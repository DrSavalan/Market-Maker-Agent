import json
import os
import tempfile
import unittest
from pathlib import Path

import rl_agent
from app import app, simulator
from rl_agent import QuoteAgent


class SimulatorApiTest(unittest.TestCase):
    def setUp(self):
        app.testing = True
        self.client = app.test_client()
        with simulator.lock:
            self.previous_agent = simulator.agent
            simulator.agent = QuoteAgent(path=None)
            simulator.market_orders.clear()
            simulator.inventory = 0
            simulator.cash = simulator.initial_cash
            simulator.realized_pnl = 0
            simulator.avg_entry = 0

    def tearDown(self):
        with simulator.lock:
            simulator.agent = self.previous_agent

    def test_state_contains_dynamic_book_and_quote(self):
        response = self.client.get('/api/state')
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(len(payload['bids']), 12)
        self.assertEqual(len(payload['asks']), 12)
        self.assertIn('buy_price', payload['recommendation'])
        self.assertIn('sell_quantity', payload['recommendation'])

    def test_market_maker_can_place_long_and_short_orders(self):
        buy = self.client.post('/api/orders', json={'side': 'buy', 'quantity': 20, 'price': 95})
        sell = self.client.post('/api/orders', json={'side': 'sell', 'quantity': 35, 'price': 105})
        self.assertEqual(buy.status_code, 201)
        self.assertEqual(sell.status_code, 201)
        payload = self.client.get('/api/state').get_json()
        self.assertEqual(len(payload['market_orders']), 2)
        self.assertTrue(any(order['side'] == 'buy' for order in payload['market_orders']))
        self.assertTrue(any(order['side'] == 'sell' for order in payload['market_orders']))

    def test_invalid_order_is_rejected(self):
        for quantity in (0, 'nan', 'inf'):
            response = self.client.post('/api/orders', json={'side': 'buy', 'quantity': quantity, 'price': 100})
            self.assertEqual(response.status_code, 400)

    def test_aggressive_short_changes_inventory_and_recommendation(self):
        before = self.client.get('/api/state').get_json()['recommendation']
        response = self.client.post('/api/orders', json={'side': 'sell', 'quantity': 100, 'price': 1})
        self.assertEqual(response.status_code, 201)
        after = response.get_json()['state']
        self.assertLess(after['inventory'], 0)
        self.assertGreater(after['recommendation']['reservation_price'], before['reservation_price'] - 1)
        self.assertTrue(after['trades'])

    def test_cancel_order(self):
        created = self.client.post('/api/orders', json={'side': 'buy', 'quantity': 25, 'price': 80}).get_json()['order']
        self.client.delete(f"/api/orders/{created['id']}")
        self.assertFalse(self.client.get('/api/state').get_json()['market_orders'])

    def test_agent_learns_and_preserves_manual_orders_when_stopped(self):
        manual = self.client.post('/api/orders', json={'side': 'buy', 'quantity': 10, 'price': 20}).get_json()['order']
        started = self.client.post('/api/agent', json={'enabled': True})
        self.assertEqual(started.status_code, 200)
        self.assertTrue(started.get_json()['agent']['enabled'])
        self.assertEqual(started.get_json()['agent']['steps'], 1)
        self.assertEqual(started.get_json()['agent']['model_status'], 'Training')
        self.assertEqual(started.get_json()['agent']['updates'], 0)
        self.assertIsNone(started.get_json()['agent']['average_reward_20'])
        with simulator.lock:
            simulator.cash += 50
        for _ in range(3):
            state = self.client.get('/api/state').get_json()
        self.assertEqual(state['agent']['steps'], 2)
        self.assertEqual(state['agent']['updates'], 1)
        self.assertEqual(state['agent']['reward_samples'], 1)
        self.assertEqual(state['agent']['average_reward_20'], state['agent']['last_reward'])
        self.assertIn(state['agent']['decision_mode'], ('Explore', 'Exploit'))
        self.assertEqual(len(state['agent']['state'].split(',')), 5)
        self.assertGreater(state['agent']['states_learned'], 0)
        self.assertTrue(any(abs(value) > 0 for row in simulator.agent.values.values() for value in row))
        self.assertIsNotNone(state['agent']['action'])
        stopped = self.client.post('/api/agent', json={'enabled': False}).get_json()
        self.assertFalse(stopped['agent']['enabled'])
        self.assertEqual(stopped['agent']['model_status'], 'Paused')
        self.assertEqual([order['id'] for order in stopped['market_orders']], [manual['id']])

    def test_agent_action_adjusts_reference_quote(self):
        with simulator.lock:
            base = simulator.agent._quote(simulator, 0)
            shift = simulator.agent._quote(simulator, 1)
            wide = simulator.agent._quote(simulator, 3)
            buy_heavy = simulator.agent._quote(simulator, 5)
        self.assertGreater(shift['reservation_price'], base['reservation_price'])
        self.assertGreater(wide['delta'], base['delta'])
        self.assertGreater(buy_heavy['buy_quantity'], base['buy_quantity'])
        self.assertLess(buy_heavy['sell_quantity'], base['sell_quantity'])

    def test_agent_rejects_invalid_control(self):
        self.assertEqual(self.client.post('/api/agent', json={'enabled': 'yes'}).status_code, 400)

    def test_agent_policy_survives_restart(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            path = str(Path(directory) / 'policy.json')
            agent = QuoteAgent(path)
            agent.values['0,0,0,0,0'] = [1.0] * 7
            agent.steps = 23
            agent._save()
            restored = QuoteAgent(path)
            self.assertEqual(restored.values, agent.values)
            self.assertEqual(restored.steps, 23)
            self.assertEqual(restored.saved_steps, 23)
            self.assertTrue(restored.loaded_from_disk)

    def test_default_policy_location_ignores_working_directory(self):
        project_dir = Path(rl_agent.__file__).resolve().parent
        with tempfile.TemporaryDirectory(dir=project_dir) as directory:
            previous_directory = Path.cwd()
            try:
                os.chdir(directory)
                self.assertEqual(QuoteAgent().path, project_dir / 'rl_policy.json')
            finally:
                os.chdir(previous_directory)

    def test_state_endpoint_saves_policy_after_twenty_decisions(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            path = Path(directory) / 'policy.json'
            with simulator.lock:
                simulator.agent = QuoteAgent(path)
            self.assertEqual(self.client.post('/api/agent', json={'enabled': True}).status_code, 200)
            for _ in range(57):
                response = self.client.get('/api/state')
                self.assertEqual(response.status_code, 200)
            self.assertEqual(response.get_json()['agent']['saved_steps'], 20)
            self.assertEqual(json.loads(path.read_text(encoding='utf-8'))['steps'], 20)
            self.assertEqual(self.client.post('/api/agent', json={'enabled': False}).status_code, 200)

    def test_dashboard_and_api_labels_are_english(self):
        dashboard = self.client.get('/')
        self.assertEqual(dashboard.status_code, 200)
        html = dashboard.get_data(as_text=True)
        self.assertIn('lang="en" dir="ltr"', html)
        self.assertIn('How the agent learns', html)
        self.assertIn('id="agentAverage"', html)
        self.assertIn('id="agentStatus"', html)
        self.assertIn('id="agentSaved"', html)
        self.assertEqual(self.client.post('/api/agent', json={'enabled': 'invalid'}).get_json()['error'], 'Invalid agent state')
        self.assertEqual(self.client.post('/api/orders', json={'side': 'buy', 'quantity': 0, 'price': 100}).get_json()['error'], 'Invalid order parameters')


if __name__ == '__main__':
    unittest.main()
