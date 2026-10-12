from __future__ import annotations

import json
import math
import os
import random
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app import OrderBookSimulator


ACTIONS = (
    ("A-S baseline", 0, 1.0, 1.0, 1.0),
    ("Raise reservation price", 1, 1.0, 1.0, 1.0),
    ("Lower reservation price", -1, 1.0, 1.0, 1.0),
    ("Widen spread", 0, 1.5, 1.0, 1.0),
    ("Tighten spread", 0, 0.6, 1.0, 1.0),
    ("Increase buy size", 0, 1.0, 1.5, 0.6),
    ("Increase sell size", 0, 1.0, 0.6, 1.5),
)


class QuoteAgent:
    def __init__(self, path: str | Path | None = Path(__file__).resolve().parent / "rl_policy.json") -> None:
        self.path = Path(path) if path else None
        self.values: dict[str, list[float]] = {}
        self.loaded_from_disk = bool(self.path and self.path.exists())
        if self.loaded_from_disk:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.values = data["values"]
            self.steps = data["steps"]
        else:
            self.steps = 0
        self.saved_steps = self.steps if self.loaded_from_disk else 0
        self.updates = 0
        self.recent_rewards: deque[float] = deque(maxlen=20)
        self.last_state: str | None = None
        self.decision_mode: str | None = None
        self.enabled = False
        self.pending: tuple[str, int, float] | None = None
        self.order_ids: set[int] = set()
        self.ticks = 0
        self.last_reward = 0.0
        self.last_action: int | None = None
        self.last_quote: dict[str, float] | None = None
        self.random = random.Random()

    def _save(self) -> None:
        if self.path is None:
            return
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps({"values": self.values, "steps": self.steps}), encoding="utf-8")
        os.replace(temporary, self.path)
        self.saved_steps = self.steps

    def set_enabled(self, market: OrderBookSimulator, enabled: bool) -> None:
        if not enabled and self.enabled:
            market.market_orders = [order for order in market.market_orders if order.order_id not in self.order_ids]
            self.order_ids.clear()
            self.pending = None
            self._save()
        self.enabled = enabled

    def _state(self, market: OrderBookSimulator) -> str:
        inventory = -1 if market.inventory < -40 else 1 if market.inventory > 40 else 0
        recent = list(market.returns)[-5:]
        momentum = sum(recent)
        trend = -1 if momentum < -0.001 else 1 if momentum > 0.001 else 0
        volatility = int(market.volatility > 0.25)
        imbalance = -1 if market.book_imbalance < -0.15 else 1 if market.book_imbalance > 0.15 else 0
        external_bids = [o for o in market.market_orders if o.order_id not in self.order_ids and o.side == "buy"]
        external_asks = [o for o in market.market_orders if o.order_id not in self.order_ids and o.side == "sell"]
        manual_pressure = -1 if sum(o.remaining for o in external_asks) > sum(o.remaining for o in external_bids) + 50 else 1 if sum(o.remaining for o in external_bids) > sum(o.remaining for o in external_asks) + 50 else 0
        return f"{inventory},{trend},{volatility},{imbalance},{manual_pressure}"

    def _quote(self, market: OrderBookSimulator, action: int) -> dict[str, float]:
        _, shift, spread_factor, buy_factor, sell_factor = ACTIONS[action]
        reference = market._recommendation()
        reservation = reference["reservation_price"] + shift * market.mid * 0.001
        delta = reference["spread"] * spread_factor
        buy_price = max(market.tick_size, market._round_price(reservation - delta / 2))
        sell_price = max(buy_price + market.tick_size, market._round_price(reservation + delta / 2))
        buy_quantity = round(min(160.0, reference["buy_quantity"] * buy_factor, max(0, 300 - market.inventory)), 2)
        sell_quantity = round(min(160.0, reference["sell_quantity"] * sell_factor, max(0, 300 + market.inventory)), 2)
        return {"buy_price": buy_price, "sell_price": sell_price, "buy_quantity": buy_quantity, "sell_quantity": sell_quantity, "reservation_price": round(reservation, 4), "delta": round(delta, 4)}

    def advance(self, market: OrderBookSimulator) -> None:
        if not self.enabled:
            return
        self.ticks += 1
        if self.ticks % 3 != 1:
            return
        state = self._state(market)
        if self.pending:
            previous, action, previous_equity = self.pending
            equity_change = market._mark_to_market() - previous_equity
            self.last_reward = round(equity_change - abs(market.inventory) * market.volatility * 0.005, 4)
            self.recent_rewards.append(self.last_reward)
            self.updates += 1
            current = self.values.setdefault(previous, [0.0] * len(ACTIONS))
            next_values = self.values.setdefault(state, [0.0] * len(ACTIONS))
            current[action] += 0.12 * (self.last_reward + 0.92 * max(next_values) - current[action])
        self.order_ids.intersection_update(order.order_id for order in market.market_orders)
        market.market_orders = [order for order in market.market_orders if order.order_id not in self.order_ids]
        self.order_ids.clear()
        choices = self.values.setdefault(state, [0.0] * len(ACTIONS))
        epsilon = max(0.05, 0.35 * math.exp(-self.steps / 500))
        if self.random.random() < epsilon:
            action = self.random.randrange(len(ACTIONS))
            self.decision_mode = "Explore"
        else:
            action = self.random.choice([index for index, value in enumerate(choices) if value == max(choices)])
            self.decision_mode = "Exploit"
        self.last_state = state
        quote = self._quote(market, action)
        for side in ("buy", "sell"):
            quantity = quote[f"{side}_quantity"]
            if quantity > 0:
                order = market.place_order(side, quantity, quote[f"{side}_price"])
                self.order_ids.add(order["id"])
        self.last_quote = quote
        self.last_action = action
        self.pending = (state, action, market._mark_to_market())
        self.steps += 1
        if self.steps % 20 == 0:
            self._save()

    def status(self) -> dict:
        rewards = self.recent_rewards
        return {
            "enabled": self.enabled,
            "model_status": "Training" if self.enabled else "Paused" if self.steps else "Untrained",
            "steps": self.steps,
            "updates": self.updates,
            "epsilon": round(max(0.05, 0.35 * math.exp(-self.steps / 500)), 3),
            "states_learned": len(self.values),
            "last_reward": self.last_reward if self.updates else None,
            "average_reward_20": round(sum(rewards) / len(rewards), 4) if rewards else None,
            "positive_reward_rate_20": round(sum(value > 0 for value in rewards) / len(rewards), 3) if rewards else None,
            "reward_samples": len(rewards),
            "action": ACTIONS[self.last_action][0] if self.last_action is not None else None,
            "decision_mode": self.decision_mode,
            "state": self.last_state,
            "quote": self.last_quote,
            "loaded_from_disk": self.loaded_from_disk,
            "saved_steps": self.saved_steps,
            "state_variables": ["Net inventory", "5-tick momentum", "Volatility regime", "Book imbalance", "Manual-order pressure"],
            "actions": [item[0] for item in ACTIONS],
        }
