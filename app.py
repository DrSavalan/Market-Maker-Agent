from __future__ import annotations

import math
import random
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from flask import Flask, jsonify, render_template, request

from rl_agent import QuoteAgent

app = Flask(__name__)


@dataclass
class MarketOrder:
    order_id: int
    side: str
    quantity: float
    price: float
    created_at: float
    remaining: float = field(init=False)

    def __post_init__(self) -> None:
        self.remaining = self.quantity


class OrderBookSimulator:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.random = random.Random()
        self.mid = 100.0
        self.tick_size = 0.01
        self.volatility = 0.22
        self.returns: deque[float] = deque(maxlen=120)
        self.price_history: deque[dict[str, float]] = deque([{"time": time.time(), "price": self.mid}], maxlen=1800)
        self.market_orders: list[MarketOrder] = []
        self.next_order_id = 1
        self.inventory = 0.0
        self.cash = 1_000_000.0
        self.initial_cash = self.cash
        self.realized_pnl = 0.0
        self.avg_entry = 0.0
        self.last_tick = time.monotonic()
        self.last_trade: dict[str, Any] | None = None
        self.trades: deque[dict[str, Any]] = deque(maxlen=30)
        self.agent = QuoteAgent()
        self.book_imbalance = 0.0

    def _round_price(self, price: float) -> float:
        return round(price / self.tick_size) * self.tick_size

    def _external_book(self) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
        spread = max(self.tick_size, self.mid * (0.00035 + self.volatility * 0.00012))
        best_bid = self._round_price(self.mid - spread / 2)
        best_ask = self._round_price(self.mid + spread / 2)
        bids = []
        asks = []
        for level in range(12):
            distance = level * self.tick_size * 2
            bids.append({"price": self._round_price(best_bid - distance), "quantity": round(self.random.uniform(8, 80) * (1 + level * 0.08), 2), "source": "random"})
            asks.append({"price": self._round_price(best_ask + distance), "quantity": round(self.random.uniform(8, 80) * (1 + level * 0.08), 2), "source": "random"})

        for order in self.market_orders:
            if order.remaining <= 0:
                continue
            row = {"price": order.price, "quantity": round(order.remaining, 2), "source": "market_maker", "order_id": order.order_id}
            (bids if order.side == "buy" else asks).append(row)
        bids.sort(key=lambda row: row["price"], reverse=True)
        asks.sort(key=lambda row: row["price"])
        return bids, asks

    def _mark_to_market(self) -> float:
        return self.cash + self.inventory * self.mid

    def _record_fill(self, order: MarketOrder, quantity: float, price: float) -> None:
        signed_quantity = quantity if order.side == "buy" else -quantity
        previous_inventory = self.inventory
        self.cash -= signed_quantity * price
        self.inventory += signed_quantity
        if previous_inventory == 0 or (previous_inventory > 0 and self.inventory > previous_inventory) or (previous_inventory < 0 and self.inventory < previous_inventory):
            total_abs = abs(previous_inventory) + abs(signed_quantity)
            self.avg_entry = ((abs(previous_inventory) * self.avg_entry) + (abs(signed_quantity) * price)) / total_abs if total_abs else 0
        else:
            closed = min(abs(previous_inventory), abs(signed_quantity))
            self.realized_pnl += closed * ((price - self.avg_entry) if previous_inventory > 0 else (self.avg_entry - price))
            if self.inventory == 0:
                self.avg_entry = 0
            elif (previous_inventory > 0) != (self.inventory > 0):
                self.avg_entry = price
        self.trades.appendleft({"time": time.strftime("%H:%M:%S"), "side": order.side, "quantity": round(quantity, 4), "price": round(price, 4), "source": "Market maker"})
        self.last_trade = self.trades[0]

    def _match_orders(self, bids: list[dict[str, float]], asks: list[dict[str, float]]) -> None:
        external_bids = [row for row in bids if row["source"] == "random"]
        external_asks = [row for row in asks if row["source"] == "random"]
        best_bid = external_bids[0]["price"] if external_bids else self.mid
        best_ask = external_asks[0]["price"] if external_asks else self.mid
        for order in self.market_orders:
            if order.remaining <= 0:
                continue
            if order.side == "buy" and external_asks and order.price >= best_ask:
                fill = min(order.remaining, external_asks[0]["quantity"] * 0.35)
                self._record_fill(order, fill, best_ask)
                order.remaining -= fill
            elif order.side == "sell" and external_bids and order.price <= best_bid:
                fill = min(order.remaining, external_bids[0]["quantity"] * 0.35)
                self._record_fill(order, fill, best_bid)
                order.remaining -= fill
            else:
                distance = abs(order.price - (best_bid if order.side == "buy" else best_ask))
                probability = 0.28 * math.exp(-distance / max(self.mid * 0.004, self.tick_size))
                if self.random.random() < probability:
                    fill = min(order.remaining, self.random.uniform(1, 12))
                    self._record_fill(order, fill, order.price)
                    order.remaining -= fill
        self.market_orders = [order for order in self.market_orders if order.remaining > 0.0001]

    def _recommendation(self) -> dict[str, float]:
        sigma = max(self.volatility, 0.01)
        gamma = 0.12
        k = 1.35
        tau = 1.0
        reservation = self.mid - gamma * self.inventory * (sigma**2) * tau
        spread = gamma * (sigma**2) * tau + (2 / gamma) * math.log(1 + gamma / k)
        spread = max(spread, self.mid * 0.0008)
        buy_price = max(self.tick_size, self._round_price(reservation - spread / 2))
        sell_price = max(buy_price + self.tick_size, self._round_price(reservation + spread / 2))
        base_quantity = max(1.0, 120.0 * (1 - min(abs(self.inventory) / 1000, 0.7)))
        buy_quantity = base_quantity * (1.0 if self.inventory <= 0 else 0.55)
        sell_quantity = base_quantity * (1.0 if self.inventory >= 0 else 0.55)
        return {"buy_price": buy_price, "sell_price": sell_price, "buy_quantity": round(buy_quantity, 2), "sell_quantity": round(sell_quantity, 2), "reservation_price": round(reservation, 4), "spread": round(spread, 4), "sigma": round(sigma, 4), "gamma": gamma}

    def tick(self) -> None:
        now = time.monotonic()
        elapsed = min(max(now - self.last_tick, 0.05), 2.0)
        self.last_tick = now
        shock = self.random.gauss(0, self.volatility * math.sqrt(elapsed) * 0.002)
        bids, asks = self._external_book()
        imbalance = (sum(row["quantity"] for row in bids[:5]) - sum(row["quantity"] for row in asks[:5])) / max(sum(row["quantity"] for row in bids[:5]) + sum(row["quantity"] for row in asks[:5]), 1)
        self.book_imbalance = imbalance
        user_imbalance = sum(order.remaining if order.side == "buy" else -order.remaining for order in self.market_orders)
        impact = max(-0.004, min(0.004, imbalance * 0.0007 + user_imbalance / 100000))
        old_mid = self.mid
        self.mid = max(0.01, self._round_price(self.mid * (1 + shock + impact)))
        self.price_history.append({"time": time.time(), "price": self.mid})
        self.returns.append((self.mid - old_mid) / old_mid)
        if len(self.returns) > 10:
            measured = math.sqrt(sum(value * value for value in self.returns) / len(self.returns)) * math.sqrt(60)
            self.volatility = max(0.03, min(1.2, measured))
        bids, asks = self._external_book()
        self._match_orders(bids, asks)

    def snapshot(self) -> dict[str, Any]:
        self.tick()
        self.agent.advance(self)
        bids, asks = self._external_book()
        recommendation = self._recommendation()
        unrealized = self.inventory * (self.mid - self.avg_entry) if self.inventory else 0
        return {"mid": self.mid, "best_bid": bids[0]["price"], "best_ask": asks[0]["price"], "spread": round(asks[0]["price"] - bids[0]["price"], 4), "volatility": self.volatility, "inventory": self.inventory, "cash": self.cash, "equity": self._mark_to_market(), "realized_pnl": self.realized_pnl, "unrealized_pnl": unrealized, "avg_entry": self.avg_entry, "bids": bids, "asks": asks, "market_orders": [{"id": order.order_id, "side": order.side, "price": order.price, "remaining": order.remaining} for order in self.market_orders], "trades": list(self.trades), "recommendation": recommendation, "price_history": list(self.price_history), "agent": self.agent.status(), "updated_at": time.time()}

    def place_order(self, side: str, quantity: float, price: float) -> dict[str, Any]:
        if side not in {"buy", "sell"} or not math.isfinite(quantity) or not math.isfinite(price) or quantity <= 0 or price < self.tick_size:
            raise ValueError("Invalid order parameters")
        order = MarketOrder(self.next_order_id, side, quantity, self._round_price(price), time.time())
        self.next_order_id += 1
        self.market_orders.append(order)
        return {"id": order.order_id, "side": order.side, "quantity": order.quantity, "price": order.price}

    def cancel_order(self, order_id: int) -> None:
        self.market_orders = [order for order in self.market_orders if order.order_id != order_id]


simulator = OrderBookSimulator()


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/state")
def state():
    with simulator.lock:
        return jsonify(simulator.snapshot())


@app.post("/api/orders")
def create_order():
    payload = request.get_json(silent=True) or {}
    try:
        with simulator.lock:
            created = simulator.place_order(str(payload.get("side", "")), float(payload.get("quantity", 0)), float(payload.get("price", 0)))
            snapshot = simulator.snapshot()
        return jsonify({"order": created, "state": snapshot}), 201
    except (TypeError, ValueError) as error:
        return jsonify({"error": str(error)}), 400


@app.delete("/api/orders/<int:order_id>")
def cancel_order(order_id: int):
    with simulator.lock:
        simulator.cancel_order(order_id)
    return jsonify({"ok": True})


@app.post("/api/agent")
def control_agent():
    payload = request.get_json(silent=True) or {}
    if payload.get("enabled") not in (True, False):
        return jsonify({"error": "Invalid agent state"}), 400
    with simulator.lock:
        simulator.agent.set_enabled(simulator, payload["enabled"])
        return jsonify(simulator.snapshot())


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
