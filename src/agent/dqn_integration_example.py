                            
                                               
from agent.dqn_agent import DQNAgent
from agent.jump_policy import JumpActions
from agent.content_policy import ContentActions
from agent.joint_action_space import JointActionSpace
import numpy as np

                                                                 
jump_actions = [JumpActions.STAY, JumpActions.BACKTRACK, (JumpActions.GOTO, 'T1_LO1'), (JumpActions.REVISIT, 'T1_LO2'), JumpActions.NULL]
content_actions = ['Q1', 'Q2', ContentActions.HINT, ContentActions.WORKED_EXAMPLE]

                          
joint_space = JointActionSpace(jump_actions, content_actions)

                 
state_dim = 128                             
action_dim = joint_space.size()
agent = DQNAgent(state_dim, action_dim)

                         
state_vec = np.random.rand(state_dim)
jump_mask = {a: 1 for a in jump_actions}                                         
content_mask = {a: 1 for a in content_actions}                                         
joint_mask = joint_space.mask(jump_mask, content_mask)

               
action_index = agent.select_action(state_vec, mask=joint_mask)
jump_action, content_action = joint_space.get_action(action_index)

print(f"Selected joint action: Jump={jump_action}, Content={content_action}")

                                                                  
                                           
                                                    
