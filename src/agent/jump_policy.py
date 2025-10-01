# jump_policy.py
# Policy for curriculum jumps (π_jump)
import numpy as np

class JumpPolicy:
    def __init__(self, action_space):
        self.action_space = action_space  # e.g., ['stay', 'backtrack', 'revisit', 'goto', 'group_work', 'NULL']

    def select_action(self, state_vec, mask=None):
        # For now, random masked action selection (placeholder for RL model)
        available = self.action_space if mask is None else [a for a, m in zip(self.action_space, mask) if m]
        return np.random.choice(available)
