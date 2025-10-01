# Add src to sys.path for module resolution
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
# train_dqn.py
# Full training loop for DQN agent across 4 environments
import numpy as np
import csv
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from agent.dqn_agent import DQNAgent
from agent.state_encoder import StateEncoder
from agent.joint_action_space import JointActionSpace
from env.rl_environment import RLEnvironment
from teacher.reward_model import RewardModel

# Placeholder: load students, curriculum, resources, question_bank for each env
# In practice, load from CSVs or generate as needed

def load_env_data(env_id):
    # Replace with real data loading
    student_latents = [np.random.rand(7) for _ in range(1000)]
    curriculum_csv = f"csv/curriculum.csv"
    resource_map = {f"T{t}_LO{lo}": [f"Q{t}_{lo}_{i}" for i in range(8)] for t in range(1, 201) for lo in range(1, 11)}
    question_bank = {qid: {'irt_difficulty': np.random.uniform(-2, 2)} for qids in resource_map.values() for qid in qids}
    return student_latents, curriculum_csv, resource_map, question_bank

num_episodes = 10  # For demo; increase for real training
max_steps = 50
results_log = []

for env_id in ['SchoolA', 'SchoolB', 'SchoolC', 'SchoolD']:
    print(f"Training in {env_id}")
    student_latents, curriculum_csv, resource_map, question_bank = load_env_data(env_id)
    env = RLEnvironment(student_latents, curriculum_csv, resource_map, question_bank)
    encoder = StateEncoder(L=2000)  # Placeholder L
    # Build action space for first state
    env.reset()
    jump_actions = env.available_jump_actions()
    content_actions = env.available_content_actions()
    joint_space = JointActionSpace(jump_actions, content_actions)
    agent = DQNAgent(state_dim=encoder.L, action_dim=joint_space.size())
    teacher = RewardModel()

    for ep in range(num_episodes):
        state = env.reset()
        total_reward = 0
        for step in range(max_steps):
            # Build state vector
            state_vec = encoder.encode({
                'mastery_vec': np.zeros(encoder.L),  # Replace with real features
            })
            # Masking
            jump_actions = env.available_jump_actions()
            content_actions = env.available_content_actions()
            joint_space = JointActionSpace(jump_actions, content_actions)
            mask = [1] * joint_space.size()  # Replace with real masking logic
            # Select action
            action_idx = agent.select_action(state_vec, mask=mask)
            jump_action, content_action = joint_space.get_action(action_idx)
            # Step
            next_state, reward, done, info = env.step(jump_action, content_action)
            # Reward shaping (placeholder)
            r_student = reward
            r_jump = 0  # Replace with teacher.r_jump(...)
            r_content = 0  # Replace with teacher.r_content(...)
            shaped_reward = teacher.total_reward(r_student, r_jump, r_content)
            # Store experience
            next_state_vec = encoder.encode({
                'mastery_vec': np.zeros(encoder.L),  # Replace with real features
            })
            agent.store(state_vec, action_idx, shaped_reward, next_state_vec, done)
            agent.update()
            total_reward += shaped_reward
            state = next_state
            if done:
                break
        agent.update_target()
        results_log.append([env_id, ep, total_reward])
        print(f"Env: {env_id}, Episode: {ep}, Total Reward: {total_reward}")

# Log results to CSV
with open('csv/dqn_results.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['env_id', 'episode', 'total_reward'])
    writer.writerows(results_log)
print("Training complete. Results saved to csv/dqn_results.csv")
