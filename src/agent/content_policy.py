# content_policy.py
# Policy for content/resource selection (π_content)
import numpy as np

class ContentPolicy:
    def __init__(self, resource_space):
        self.resource_space = resource_space  # list of resource IDs or objects

    def select_action(self, state_vec, mask=None):
        # For now, random masked action selection (placeholder for RL model)
        available = self.resource_space if mask is None else [r for r, m in zip(self.resource_space, mask) if m]
        return np.random.choice(available)
