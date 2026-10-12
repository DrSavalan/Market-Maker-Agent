const $ = (id) => document.getElementById(id);
let state = null;
const history = [];
const fmt = (value, digits = 2) => Number(value).toLocaleString('en-US', {maximumFractionDigits: digits, minimumFractionDigits: digits});
const signed = (value) => `${value >= 0 ? '+' : ''}${fmt(value)}`;

function renderBook(side, rows) {
  const el = $(side);
  el.innerHTML = rows.map(row => `<div class="book-row" style="--fill:${row.source === 'market_maker' ? '#75a7ff' : side === 'asks' ? '#fb7185' : '#36d39f'}"><span class="qty">${fmt(row.quantity)}</span><span class="price">${fmt(row.price)}</span></div>`).join('');
}
function render(stateData) {
  state = stateData;
  $('mid').textContent = fmt(state.mid);
  $('spread').textContent = `Spread ${fmt(state.spread, 4)}`;
  $('inventory').textContent = signed(state.inventory);
  $('equity').textContent = fmt(state.equity);
  $('cash').textContent = `Cash ${fmt(state.cash)}`;
  $('unrealized').textContent = signed(state.unrealized_pnl);
  $('realized').textContent = `Realized ${signed(state.realized_pnl)}`;
  $('volatility').textContent = `Volatility ${fmt(state.volatility * 100, 2)}%`;
  $('updated').textContent = new Date().toLocaleTimeString('en-US');
  const agent = state.agent;
  $('agentStatus').textContent = agent.model_status;
  $('agentSteps').textContent = agent.steps.toLocaleString('en-US');
  $('agentUpdates').textContent = agent.updates.toLocaleString('en-US');
  $('agentStates').textContent = agent.states_learned.toLocaleString('en-US');
  $('agentReward').textContent = agent.last_reward === null ? '—' : signed(agent.last_reward);
  $('agentAverage').textContent = agent.average_reward_20 === null ? '—' : signed(agent.average_reward_20);
  $('agentPositive').textContent = agent.positive_reward_rate_20 === null ? '—' : `${fmt(agent.positive_reward_rate_20 * 100, 1)}% of ${agent.reward_samples}`;
  $('agentSaved').textContent = agent.saved_steps ? agent.saved_steps.toLocaleString('en-US') : 'Not yet';
  $('agentEpsilon').textContent = `${fmt(agent.epsilon * 100, 1)}%`;
  $('agentAction').textContent = agent.action || 'No decision yet';
  $('agentMode').textContent = agent.decision_mode || '—';
  $('agentQuote').textContent = agent.quote ? `Bid ${fmt(agent.quote.buy_quantity)} @ ${fmt(agent.quote.buy_price)} · Ask ${fmt(agent.quote.sell_quantity)} @ ${fmt(agent.quote.sell_price)}` : '—';
  $('agentState').textContent = agent.state ? `State (inventory, trend, volatility, book, manual): ${agent.state}` : 'State: —';
  $('agentStorage').textContent = agent.loaded_from_disk ? 'Saved policy loaded at startup.' : agent.saved_steps ? 'Policy saved during this session.' : 'No saved policy loaded.';
  $('toggleAgent').textContent = agent.enabled ? 'Stop agent & cancel its orders' : 'Start training';
  renderBook('asks', state.asks.slice().reverse());
  renderBook('bids', state.bids);
  const r = state.recommendation;
  $('suggestBuyPrice').textContent = fmt(r.buy_price);
  $('suggestBuyQty').textContent = `${fmt(r.buy_quantity)} units`;
  $('suggestSellPrice').textContent = fmt(r.sell_price);
  $('suggestSellQty').textContent = `${fmt(r.sell_quantity)} units`;
  $('reservation').textContent = fmt(r.reservation_price);
  $('modelSpread').textContent = fmt(r.spread, 4);
  $('orderCount').textContent = state.market_orders.length.toLocaleString('en-US');
  $('myOrders').innerHTML = state.market_orders.length ? state.market_orders.map(o => `<div class="order-line"><b class="${o.side}">${o.side === 'buy' ? 'Buy' : 'Sell'} #${o.id}</b><span>${fmt(o.remaining)} × ${fmt(o.price)}</span><button class="cancel" onclick="cancelOrder(${o.id})">Cancel</button></div>`).join('') : '<p class="empty">No open orders yet.</p>';
  $('trades').innerHTML = state.trades.length ? state.trades.slice(0, 8).map(t => `<div class="trade"><span>${t.time} · <b class="${t.side}">${t.side === 'buy' ? 'Buy' : 'Sell'}</b></span><span>${fmt(t.quantity)} @ ${fmt(t.price)}</span></div>`).join('') : '<p class="empty">Waiting for a fill…</p>';
  history.push(state.mid); if (history.length > 80) history.shift(); drawChart();
}
async function load() { try { const response = await fetch('/api/state'); render(await response.json()); } catch (_) { $('updated').textContent = 'Disconnected'; } }
async function cancelOrder(id) { await fetch(`/api/orders/${id}`, {method: 'DELETE'}); load(); }
window.cancelOrder = cancelOrder;

document.querySelectorAll('[data-side]').forEach(button => button.addEventListener('click', () => { document.querySelectorAll('[data-side]').forEach(item => item.classList.remove('active')); button.classList.add('active'); $('side').value = button.dataset.side; }));
$('refreshQuote').addEventListener('click', load);
$('toggleAgent').addEventListener('click', async () => {
  const response = await fetch('/api/agent', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({enabled: !state.agent.enabled})});
  if (response.ok) render(await response.json());
});
$('useQuote').addEventListener('click', () => { const side = $('side').value; $('price').value = side === 'buy' ? state.recommendation.buy_price : state.recommendation.sell_price; $('quantity').value = side === 'buy' ? state.recommendation.buy_quantity : state.recommendation.sell_quantity; });
$('orderForm').addEventListener('submit', async (event) => { event.preventDefault(); const response = await fetch('/api/orders', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({side:$('side').value, price:$('price').value, quantity:$('quantity').value})}); const result = await response.json(); $('formMessage').textContent = response.ok ? `Order #${result.order.id} added to the book.` : result.error; if (response.ok) render(result.state); });
function drawChart() { const canvas = $('chart'), context = canvas.getContext('2d'), width = canvas.clientWidth, height = canvas.height; const ratio = window.devicePixelRatio || 1; canvas.width = width * ratio; canvas.height = height * ratio; context.scale(ratio, ratio); context.clearRect(0, 0, width, height); if (history.length < 2) return; const min = Math.min(...history), max = Math.max(...history), range = max - min || 1; context.strokeStyle = '#263955'; context.lineWidth = 1; for (let i = 1; i < 4; i++) { const y = height * i / 4; context.beginPath(); context.moveTo(0, y); context.lineTo(width, y); context.stroke(); } context.strokeStyle = '#75a7ff'; context.lineWidth = 2; context.beginPath(); history.forEach((value, index) => { const x = index * width / (history.length - 1); const y = height - ((value - min) / range) * (height - 20) - 10; index ? context.lineTo(x, y) : context.moveTo(x, y); }); context.stroke(); }
load(); setInterval(load, 1200); window.addEventListener('resize', drawChart);
