                                           
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
              
                                                        
import numpy as np
import csv
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from agent.dqn_agent import TwoHeadDQNAgent
from agent.state_encoder import StateEncoder
from agent.joint_action_space import JointActionSpace
from env.rl_environment import RLEnvironment
from teacher.reward_model import RewardModel

                                                                               
                                                   

def load_env_data(env_id):
                                    
    student_latents = [np.random.rand(7) for _ in range(1000)]
    curriculum_csv = f"csv/curriculum.csv"
    resource_map = {f"T{t}_LO{lo}": [f"Q{t}_{lo}_{i}" for i in range(8)] for t in range(1, 201) for lo in range(1, 11)}
    question_bank = {qid: {'irt_difficulty': np.random.uniform(-2, 2)} for qids in resource_map.values() for qid in qids}
    return student_latents, curriculum_csv, resource_map, question_bank

num_episodes = 10                                        
max_steps = 50
results_log = []

                                
intervention_log = []
intervention_log_header = ['env_id', 'episode', 'step', 'student_id', 'event', 'details']

                                          
trajectory_log = []
trajectory_log_header = ['env_id', 'episode', 'step', 'student_id', 'state', 'action_idx', 'jump_action', 'content_action', 'reward', 'next_state', 'done', 'info']

for env_id in ['SchoolA', 'SchoolB', 'SchoolC', 'SchoolD']:
    print(f"Training in {env_id}")
    student_latents, curriculum_csv, resource_map, question_bank = load_env_data(env_id)
    env = RLEnvironment(student_latents, curriculum_csv, resource_map, question_bank)
    encoder = StateEncoder(L=2000)                 
                                                               
    env.reset()
    jump_actions = env.available_jump_actions()
    content_actions = env.available_content_actions()
    joint_space = JointActionSpace(jump_actions, content_actions)
                                                    
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
    agent = TwoHeadDQNAgent(
        state_dim=sample_state_vec.shape[0],
        jump_dim=len(jump_actions),
        content_dim=len(content_actions)
    )
    teacher = RewardModel()

    for ep in range(num_episodes):
        state = env.reset()
        total_reward = 0
                                                                  
        student_id = getattr(env, 'current_student_id', None)
        if student_id is None and isinstance(state, dict):
            student_id = state.get('student_id', None)
    failure_count = 0
    mastery_achieved = set()
    for step in range(max_steps):
                                                                    
            env_state = env._get_state()
            mastery_vec = np.zeros(encoder.L)
            mastery_vel = np.zeros(3*encoder.L)
            latents = np.array(env_state.get('latents', np.zeros(7)))
                                                                                
            mastery_dict = env_state.get('mastery', {})
            lo_ids = list(mastery_dict.keys())
            for i, lo in enumerate(lo_ids):
                if i < encoder.L:
                    mastery_vec[i] = mastery_dict[lo]
                                                                                                                                
                                                                      
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
                                                                 
            jump_actions = env.available_jump_actions()
            content_actions = env.available_content_actions()
            joint_space = JointActionSpace(jump_actions, content_actions)
                                                             
            jump_mask = [int(j in jump_actions) for j in jump_actions]
            content_mask = [int(c in content_actions) for c in content_actions]
                                                 
            jump_idx = agent.select_jump_action(state_vec, mask=jump_mask)
            content_idx = agent.select_content_action(state_vec, mask=content_mask)
            jump_action = jump_actions[jump_idx]
            content_action = content_actions[content_idx]
                  
            next_state, reward, done, info = env.step(jump_action, content_action)
            print(f"Step: {step}, Done: {done}")
                                
                                                     
            correct = info.get('correct', reward > 0)
            if not correct:
                failure_count += 1
            else:
                failure_count = 0
            if failure_count == 8:
                intervention_log.append([
                    env_id, ep, step, student_id, 'instructor_flag', '8 consecutive failures'
                ])
                                                                      
            mastery_dict = next_state.get('mastery', {}) if isinstance(next_state, dict) else {}
            for lo, m in mastery_dict.items():
                if m >= 0.8 and lo not in mastery_achieved:
                    mastery_achieved.add(lo)
                    intervention_log.append([
                        env_id, ep, step, student_id, 'mastery_achieved', f'LO {lo} mastery {m:.2f}'
                    ])
                                                         
            correct = info.get('correct', reward > 0)
            mastery_gain = info.get('mastery_gain', 0)
            time_penalty = 0                                                      
            slip = 0                                                          
            guess = 0                                                          
            r_student = teacher.r_student(correct, mastery_gain, time_penalty, slip, guess)
                                                                       
            jump_info = {
                'prereq_safe': True,                                                
                'zpd': False,                                                    
                'coverage': 0,                                                     
                'stale': 0,                                                                 
                'retention': False,                                               
                'overconfident': False,                                                     
                'jump_vel': 0,                                                        
                'v_max': 1                                               
            }
            r_jump = teacher.r_jump(jump_info)
            content_info = {
                'appropriate_difficulty': True,                                             
                'modality_switch': 0,                                                          
                'type_imbalance': 0,                                                       
                'scaffold': False,                                                           
                'hint_on_easy': False,                                                        
                'overexposed': 0                                                             
            }
            r_content = teacher.r_content(content_info)
            shaped_reward = teacher.total_reward(r_student, r_jump, r_content)
                              
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
            agent.store(state_vec, jump_idx, content_idx, shaped_reward, next_state_vec, done)
            agent.update()
                                              
            trajectory_log.append([
                env_id,
                ep,
                step,
                student_id,
                state_vec.tolist(),
                f"jump:{jump_idx}|content:{content_idx}",
                jump_action,
                content_action,
                shaped_reward,
                next_state_vec.tolist(),
                done,
                info
            ])
            print(f"Trajectory log length after append: {len(trajectory_log)}")
            total_reward += shaped_reward
            state = next_state
            if done:
                break
    agent.update_target()
    results_log.append([env_id, ep, total_reward])
    print(f"Env: {env_id}, Episode: {ep}, Total Reward: {total_reward}")

                    
with open('csv/dqn_results.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['env_id', 'episode', 'total_reward'])
    writer.writerows(results_log)
print("Training complete. Results saved to csv/dqn_results.csv")

                                
with open('csv/rl_interventions.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(intervention_log_header)
    writer.writerows(intervention_log)
print("Intervention events saved to csv/rl_interventions.csv")

                                          
import os
csv_path = os.path.abspath('csv/rl_student_trajectories.csv')
print(f"Trajectory log length before writing CSV: {len(trajectory_log)}")
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(trajectory_log_header)
    writer.writerows(trajectory_log)
    f.flush()
    os.fsync(f.fileno())
    print(f"Detailed trajectories saved to {csv_path}")
