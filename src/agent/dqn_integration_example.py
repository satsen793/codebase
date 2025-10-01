# dqn_integration_example.py
# Example: Using JointActionSpace with DQNAgent
from agent.dqn_agent import DQNAgent
from agent.jump_policy import JumpActions
from agent.content_policy import ContentActions
from agent.joint_action_space import JointActionSpace
import numpy as np

# Example jump and content actions (normally generated per state)
jump_actions = [JumpActions.STAY, JumpActions.BACKTRACK, (JumpActions.GOTO, 'T1_LO1'), (JumpActions.REVISIT, 'T1_LO2'), JumpActions.NULL]
content_actions = ['Q1', 'Q2', ContentActions.HINT, ContentActions.WORKED_EXAMPLE]

# Build joint action space
joint_space = JointActionSpace(jump_actions, content_actions)

# DQN agent setup
state_dim = 128  # Example state vector size
action_dim = joint_space.size()
agent = DQNAgent(state_dim, action_dim)

# Example state and masks
state_vec = np.random.rand(state_dim)
jump_mask = {a: 1 for a in jump_actions}  # All allowed (replace with real logic)
content_mask = {a: 1 for a in content_actions}  # All allowed (replace with real logic)
joint_mask = joint_space.mask(jump_mask, content_mask)

# Select action
action_index = agent.select_action(state_vec, mask=joint_mask)
jump_action, content_action = joint_space.get_action(action_index)

print(f"Selected joint action: Jump={jump_action}, Content={content_action}")

# When stepping the environment, use (jump_action, content_action)
# When storing experience, use action_index
# When updating, use action_dim = joint_space.size()
