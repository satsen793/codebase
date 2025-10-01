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

# For intervention event logging
intervention_log = []
intervention_log_header = ['env_id', 'episode', 'step', 'student_id', 'event', 'details']

# For detailed per-step trajectory logging
trajectory_log = []
trajectory_log_header = ['env_id', 'episode', 'step', 'student_id', 'state', 'action_idx', 'jump_action', 'content_action', 'reward', 'next_state', 'done', 'info']

for env_id in ['SchoolA', 'SchoolB', 'SchoolC', 'SchoolD']:
    print(f"Training in {env_id}")
    student_latents, curriculum_csv, resource_map, question_bank = load_env_data(env_id)
    env = RLEnvironment(student_latents, curriculum_csv, resource_map, question_bank)
    encoder = StateEncoder(L=2000)  # Placeholder L
    # Build joint action space ONCE per episode (fixed mapping)
    env.reset()
    jump_actions = env.available_jump_actions()
    content_actions = env.available_content_actions()
    joint_space = JointActionSpace(jump_actions, content_actions)
    # Get true state vector size from a sample state
    sample_state = env.reset()
    sample_state_vec = encoder.encode({
        'mastery_vec': np.zeros(encoder.L),
        'mastery_vel': np.zeros(3*encoder.L),
        'latents': np.zeros(7),
        'uncertainty': np.zeros(7),
        'error_embed': np.zeros(32),
        'hist_embed': np.zeros(64),
        'curriculum_pos': [0],
        'unlocks': np.zeros(encoder.L),
        'jump_vel': 0,
        'context': np.zeros(6)
    })
    agent = DQNAgent(state_dim=sample_state_vec.shape[0], action_dim=joint_space.size())
    teacher = RewardModel()

    for ep in range(num_episodes):
        state = env.reset()
        total_reward = 0
        # Try to extract student_id from env or state if available
        student_id = getattr(env, 'current_student_id', None)
        if student_id is None and isinstance(state, dict):
            student_id = state.get('student_id', None)
    failure_count = 0
    mastery_achieved = set()
    for step in range(max_steps):
            # Build state vector from real environment/student state
            env_state = env._get_state()
            mastery_vec = np.zeros(encoder.L)
            mastery_vel = np.zeros(3*encoder.L)
            latents = np.array(env_state.get('latents', np.zeros(7)))
            # If mastery is a dict, convert to vector (assume LO ids are sorted)
            mastery_dict = env_state.get('mastery', {})
            lo_ids = list(mastery_dict.keys())
            for i, lo in enumerate(lo_ids):
                if i < encoder.L:
                    mastery_vec[i] = mastery_dict[lo]
            # Placeholder for mastery velocity, uncertainty, error_embed, hist_embed, curriculum_pos, unlocks, jump_vel, context
            # (You can expand these as your env/student supports them)
            state_vec = encoder.encode({
                'mastery_vec': mastery_vec,
                'mastery_vel': mastery_vel,
                'latents': latents,
                'uncertainty': np.zeros(7),
                'error_embed': np.zeros(encoder.error_dim),
                'hist_embed': np.zeros(encoder.hist_dim),
                'curriculum_pos': [lo_ids.index(env_state.get('current_lo', lo_ids[0])) if lo_ids else 0],
                'unlocks': np.zeros(encoder.L),
                'jump_vel': 0,
                'context': np.zeros(encoder.context_dim)
            })
            # Masking (context-aware, but using fixed joint_space)
            # Get available jump and content actions for this step
            available_jump_actions = env.available_jump_actions()
            available_content_actions = env.available_content_actions()
            # Build jump/content masks (1 if available, 0 if not)
            jump_mask = [(j in available_jump_actions) for j in joint_space.action_pairs[0][0:len(jump_actions)]]
            content_mask = [(c in available_content_actions) for c in joint_space.action_pairs[0][1:len(content_actions)]]
            # Use joint_space.mask to get the mask for the joint action space
            mask = joint_space.mask(jump_mask, content_mask)
            # Select action
            action_idx = agent.select_action(state_vec, mask=mask)
            jump_action, content_action = joint_space.get_action(action_idx)
            # Step
            next_state, reward, done, info = env.step(jump_action, content_action)
            # Intervention hooks
            # 1. Repeated failures (e.g., 8 mistakes)
            correct = info.get('correct', reward > 0)
            if not correct:
                failure_count += 1
            else:
                failure_count = 0
            if failure_count == 8:
                intervention_log.append([
                    env_id, ep, step, student_id, 'instructor_flag', '8 consecutive failures'
                ])
            # 2. Mastery achievement (e.g., mastery >= 0.8 for any LO)
            mastery_dict = next_state.get('mastery', {}) if isinstance(next_state, dict) else {}
            for lo, m in mastery_dict.items():
                if m >= 0.8 and lo not in mastery_achieved:
                    mastery_achieved.add(lo)
                    intervention_log.append([
                        env_id, ep, step, student_id, 'mastery_achieved', f'LO {lo} mastery {m:.2f}'
                    ])
            # Reward shaping (use real info if available)
            correct = info.get('correct', reward > 0)
            mastery_gain = info.get('mastery_gain', 0)
            time_penalty = 0  # Placeholder, can be computed from env if available
            slip = 0  # Placeholder, can be computed from student if available
            guess = 0  # Placeholder, can be computed from student if available
            r_student = teacher.r_student(correct, mastery_gain, time_penalty, slip, guess)
            # Build jump_info and content_info dicts for reward shaping
            jump_info = {
                'prereq_safe': True,  # Placeholder, can be computed from curriculum
                'zpd': False,         # Placeholder, can be computed from mastery
                'coverage': 0,        # Placeholder, can be incremented for new LOs
                'stale': 0,           # Placeholder, can be incremented for staying too long
                'retention': False,   # Placeholder, can be set for spaced revisit
                'overconfident': False, # Placeholder, can be set if mastery high but failed
                'jump_vel': 0,        # Placeholder, can be computed from jump history
                'v_max': 1            # Placeholder, can be set as needed
            }
            r_jump = teacher.r_jump(jump_info)
            content_info = {
                'appropriate_difficulty': True, # Placeholder, can be set from resource info
                'modality_switch': 0,           # Placeholder, can be set from modality history
                'type_imbalance': 0,            # Placeholder, can be set from type history
                'scaffold': False,              # Placeholder, can be set if scaffolding used
                'hint_on_easy': False,          # Placeholder, can be set if hint on easy item
                'overexposed': 0                # Placeholder, can be set from resource usage
            }
            r_content = teacher.r_content(content_info)
            shaped_reward = teacher.total_reward(r_student, r_jump, r_content)
            # Store experience
            next_env_state = next_state if isinstance(next_state, dict) else {}
            next_mastery_vec = np.zeros(encoder.L)
            next_mastery_dict = next_env_state.get('mastery', {})
            next_lo_ids = list(next_mastery_dict.keys())
            for i, lo in enumerate(next_lo_ids):
                if i < encoder.L:
                    next_mastery_vec[i] = next_mastery_dict[lo]
            next_state_vec = encoder.encode({
                'mastery_vec': next_mastery_vec,
                'mastery_vel': mastery_vel,
                'latents': np.array(next_env_state.get('latents', np.zeros(7))),
                'uncertainty': np.zeros(7),
                'error_embed': np.zeros(encoder.error_dim),
                'hist_embed': np.zeros(encoder.hist_dim),
                'curriculum_pos': [next_lo_ids.index(next_env_state.get('current_lo', next_lo_ids[0])) if next_lo_ids else 0],
                'unlocks': np.zeros(encoder.L),
                'jump_vel': 0,
                'context': np.zeros(encoder.context_dim)
            })
            agent.store(state_vec, action_idx, shaped_reward, next_state_vec, done)
            agent.update()
            # Log trajectory step
            trajectory_log.append([
                env_id,
                ep,
                step,
                student_id,
                state_vec.tolist(),
                action_idx,
                jump_action,
                content_action,
                shaped_reward,
                next_state_vec.tolist(),
                done,
                info
            ])
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

# Log intervention events to CSV
with open('csv/rl_interventions.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(intervention_log_header)
    writer.writerows(intervention_log)
print("Intervention events saved to csv/rl_interventions.csv")

# Log detailed student trajectories to CSV
with open('csv/rl_student_trajectories.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(trajectory_log_header)
    writer.writerows(trajectory_log)
print("Detailed trajectories saved to csv/rl_student_trajectories.csv")
