import os
import numpy
from dotenv import load_dotenv
from policy import RandomPolicy, Policy

load_dotenv()

class Policy2212387(Policy):
    def __init__(self):
        policy_id=int(os.getenv('POLICY_ID', '1'))
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        self.policy_id = policy_id
        self.q_table = {}
        self.learning_rate = float(os.getenv('LEARNING_RATE', '0.1'))
        self.discount_factor = float(os.getenv('DISCOUNT_FACTOR', '0.9'))
        self.exploration_rate = float(os.getenv('EXPLORATION_RATE', '0.2'))

    def get_action(self, observation, info):
        if self.policy_id == 1:
            return self._column_generation_action(observation, info)
        elif self.policy_id == 2:
            return self._q_learning_action(observation, info)

    def _column_generation_action(self, observation, info):
        # Lấy thông tin stock và product từ observation
        stocks = observation["stocks"]
        products = observation["products"]

        # Sắp xếp product theo diện tích giảm dần để tăng filled ratio
        sorted_products = sorted(products, key=lambda p: numpy.prod(p['size']), reverse=True)

        for product in sorted_products:
            if product["quantity"] == 0:
                continue
            for stock_idx, stock in enumerate(stocks):
                position_x, position_y = self._find_position(stock, product["size"])
                if position_x is not None and position_y is not None:
                    # Đặt product nếu có vị trí phù hợp
                    return {
                        "stock_idx": stock_idx,
                        "size": product["size"],
                        "position": (position_x, position_y)
                    }

        # Nếu không tìm thấy chính sách phù hợp, gọi chính sách ngẫu nhiên
        return RandomPolicy().get_action(observation, info)

    def _find_position(self, stock, prod_size):
        # Tìm vị trí có thể đặt product
        stock_w, stock_h = self._get_stock_size_(stock)
        prod_w, prod_h = prod_size
        for x in range(stock_w - prod_w + 1):
            for y in range(stock_h - prod_h + 1):
                if self._can_place_(stock, (x, y), prod_size):
                    return x, y
        return None, None

    def _q_learning_action(self, observation, info):
        state_key = self._generate_state_key(observation)

        # Khởi tạo Q_table nếu đây là state mới
        if state_key not in self.q_table:
            self.q_table[state_key] = {}

        # Áp dụng chiến lược epsilon-greedy để chọn hành động
        if numpy.random.rand() < self.exploration_rate:
            action = self._random_action(observation)
        else:
            action = self._select_best_action(state_key, observation)

        # Cập nhật Q_table với hành động đã chọn
        self._update_q_table(state_key, action, observation, info)
        return action

    def _random_action(self, observation):
        # Chọn một action hợp lệ bất kỳ
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

    def _select_best_action(self, state_key, observation):
        # Chọn action tốt nhất dựa trên Q_value
        actions = self.q_table[state_key]
        return max(actions, key=actions.get, default=self._random_action(observation))

    def _update_q_table(self, state_key, action, observation, info):
        # Cập nhật Q_value cho hành động đã chọn
        action_key = (action["stock_idx"], tuple(action["size"]), tuple(action["position"]))
        reward = self._calculate_reward(action, observation)
        next_state_key = self._generate_state_key(observation)

        current_q = self.q_table[state_key].get(action_key, 0)
        max_future_q = max(self.q_table.get(next_state_key, {}).values(), default=0)

        self.q_table[state_key][action_key] = (1 - self.learning_rate) * current_q + self.learning_rate * (reward + self.discount_factor * max_future_q)

    def _calculate_reward(self, action, observation):
        # Tính toán reward dựa trên diện tích đã cắt
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

    def _generate_state_key(self, observation):
        # Tạo khóa đại diện cho trạng thái hiện tại
        stocks = observation["stocks"]
        products = observation["products"]
        return str(stocks) + str(products)