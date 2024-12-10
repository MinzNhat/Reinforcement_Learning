import os
import gymnasium as gym
import gym_cutting_stock
from dotenv import load_dotenv
from student_submissions.s2212387.policy2212387 import Policy2212387

env = gym.make(
    "gym_cutting_stock/CuttingStock-v0",
    render_mode="human",
)

load_dotenv()

NUM_EPISODES = int(os.getenv('NUM_EPISODES', '100'))

if __name__ == "__main__":
    policy2211257 = Policy2212387()  # Khởi tạo chính sách

    episode = 0
    total_rewards = []
    filled_ratios = []

    try:
        while episode < NUM_EPISODES:
            # Đặt lại môi trường với seed ứng với số thứ tự episode hiện tại
            observation, info = env.reset(seed=episode)
            episode += 1

            print(f"\n--- Starting Episode {episode} ---")
            print(f"Initial products: {observation['products']}")
            print(f"Initial info: {info}")

            terminated = False  # Trạng thái kết thúc
            truncated = False  # Trạng thái dừng do vượt thời gian
            episode_reward = 0  # Tổng điểm thưởng cho episode này

            while not (terminated or truncated):
                # Lấy hành động từ chính sách
                action = policy2211257.get_action(observation, info)
                observation, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                print(f"{info}")

            # Lưu filled_ratio nếu có
            if "filled_ratio" in info:
                filled_ratios.append(info["filled_ratio"])
            else:
                filled_ratios.append(None)

            # Lưu total_rewards của episode
            total_rewards.append(episode_reward)
    except KeyboardInterrupt:
        print(f"\n--- Training interrupted ---")
    finally:
        print(f"\n--- Training Summary ---")
        if not total_rewards and not filled_ratios:
            print("No data to summarize; all episodes were interrupted.")
        else:
            print(f"Total rewards for each episode: {total_rewards}")
            print(f"Filled ratios for each episode: {filled_ratios}")
            # Tìm filled ratio cao nhất
            max_filled_ratio = max([r for r in filled_ratios if r is not None], default=None)
            if max_filled_ratio is not None:
                print(f"Highest filled ratio: {max_filled_ratio:.2f}")
            else:
                print("No valid filled ratios.")
        env.close()
