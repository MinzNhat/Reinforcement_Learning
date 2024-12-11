import os
import numpy
from dotenv import load_dotenv
from policy import RandomPolicy, Policy

load_dotenv()

class Policy2212387(Policy):
    def __init__(self):
        policy_id = int(os.getenv('POLICY_ID', '1'))
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        self.policy_id = policy_id
        self.q_table = {}
        self.sarsa_table = {}
        self.learning_rate = float(os.getenv('LEARNING_RATE', '0.1'))
        self.discount_factor = float(os.getenv('DISCOUNT_FACTOR', '0.9'))
        self.exploration_rate = float(os.getenv('EXPLORATION_RATE', '0.2'))

    def get_action(self, observation, info):
        if self.policy_id == 1:
            return self._sarsa_action(observation)
        elif self.policy_id == 2:
            return self._q_learning_action(observation)

    def _sarsa_action(self, observation):
        """SARSA-based action selection."""
        state_key = self._generate_state_key(observation)

        # Khởi tạo SARSA table nếu là trạng thái mới
        if state_key not in self.sarsa_table:
            self.sarsa_table[state_key] = {}

        # Áp dụng epsilon-greedy để chọn hành động
        if numpy.random.rand() < self.exploration_rate:
            action = self._random_action(observation)  # Truyền observation vào đây
        else:
            action = self._select_best_action_sarsa(state_key, observation)  # Truyền observation vào đây

        # Lấy phần thưởng và cập nhật SARSA table
        reward = self._calculate_reward(action, observation)
        next_state_key = self._generate_state_key(observation)
        
        # Chọn hành động tiếp theo (next_action) cho SARSA
        next_action = self._random_action(observation) if numpy.random.rand() < self.exploration_rate else self._select_best_action_sarsa(next_state_key, observation)

        # Cập nhật SARSA table
        self._update_sarsa_table(state_key, action, reward, next_state_key, next_action)
        return action

    def _update_sarsa_table(self, state_key, action, reward, next_state_key, next_action):
        """Cập nhật giá trị SARSA table."""
        action_key = (action["stock_idx"], tuple(action["size"]), tuple(action["position"]))
        next_action_key = (next_action["stock_idx"], tuple(next_action["size"]), tuple(next_action["position"]))

        current_q = self.sarsa_table[state_key].get(action_key, 0)
        next_q = self.sarsa_table.get(next_state_key, {}).get(next_action_key, 0)

        self.sarsa_table[state_key][action_key] = (1 - self.learning_rate) * current_q + self.learning_rate * (reward + self.discount_factor * next_q)

    def _select_best_action_sarsa(self, state_key, observation):
        """Chọn hành động tốt nhất dựa trên giá trị SARSA."""
        actions = self.sarsa_table[state_key]
        return max(actions, key=actions.get, default=self._random_action(observation))

    def _q_learning_action(self, observation):
        """Q-Learning-based action selection."""
        state_key = self._generate_state_key(observation)

        # Khởi tạo Q_table nếu đây là trạng thái mới
        if state_key not in self.q_table:
            self.q_table[state_key] = {}

        # Áp dụng chiến lược epsilon-greedy để chọn hành động
        if numpy.random.rand() < self.exploration_rate:
            action = self._random_action(observation)
        else:
            action = self._select_best_action_q_learning(state_key)

        # Cập nhật Q_table với hành động đã chọn
        self._update_q_table(state_key, action, observation)
        return action

    def _select_best_action_q_learning(self, state_key):
        """Chọn hành động tốt nhất dựa trên Q-value."""
        actions = self.q_table[state_key]
        return max(actions, key=actions.get, default=self._random_action())

    def _update_q_table(self, state_key, action, observation):
        """Cập nhật Q-value cho hành động đã chọn."""
        action_key = (action["stock_idx"], tuple(action["size"]), tuple(action["position"]))
        reward = self._calculate_reward(action, observation)
        next_state_key = self._generate_state_key(observation)

        current_q = self.q_table[state_key].get(action_key, 0)
        max_future_q = max(self.q_table.get(next_state_key, {}).values(), default=0)

        self.q_table[state_key][action_key] = (1 - self.learning_rate) * current_q + self.learning_rate * (reward + self.discount_factor * max_future_q)

    def _generate_state_key(self, observation):
        """Tạo khóa đại diện cho trạng thái từ observation."""
        stocks = observation["stocks"]
        products = observation["products"]
        return str(stocks) + str(products)

    def _random_action(self, observation):
        """Chọn một action hợp lệ bất kỳ."""
        for product in observation["products"]:
            if product["quantity"] > 0:
                for stock_idx, stock in enumerate(observation["stocks"]):
                    position_x, position_y = self._find_position(stock, product["size"])
                    if position_x is not None and position_y is not None:
                        return {
                            "stock_idx": stock_idx,
                            "size": product["size"],
                            "position": (position_x, position_y)
                        }
        # Nếu không có action hợp lệ, trả về giá trị mặc định
        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

    def _find_position(self, stock, prod_size):
        """Tìm vị trí có thể đặt product."""
        stock_w, stock_h = self._get_stock_size_(stock)
        prod_w, prod_h = prod_size
        for x in range(stock_w - prod_w + 1):
            for y in range(stock_h - prod_h + 1):
                if self._can_place_(stock, (x, y), prod_size):
                    return x, y
        return None, None

    def _calculate_reward(self, action, observation):
        """Tính toán reward dựa trên diện tích đã cắt."""
        stock_idx = action["stock_idx"]
        size = action["size"]
        position = action["position"]

        if stock_idx == -1:
            return -5  # Trừ điểm nếu không tìm thấy action hợp lệ

        stock = observation["stocks"][stock_idx]
        if self._can_place_(stock, position, size):
            filled_ratio = numpy.prod(size) / numpy.prod(self._get_stock_size_(stock))
            return 10 + filled_ratio * 20  # Tăng phần thưởng dựa trên diện tích đã cắt
        return -10  # Phạt nếu không thể cắt