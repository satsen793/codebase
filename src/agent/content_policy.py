# content_policy.py
# Policy for content/resource selection (π_content)
import numpy as np

# Enumerate content/meta-actions as per proj1oct.txt
class ContentActions:
    HINT = 'hint'
    WORKED_EXAMPLE = 'worked_example'
    SELF_EXPLAIN_PROMPT = 'self_explain_prompt'
    DISCUSSION_PROMPT = 'discussion_prompt'

    @staticmethod
    def meta_actions():
        return [ContentActions.HINT, ContentActions.WORKED_EXAMPLE, ContentActions.SELF_EXPLAIN_PROMPT, ContentActions.DISCUSSION_PROMPT]

class ContentPolicy:
    def __init__(self, resource_space):
        # resource_space: list of resource IDs or objects for current LO
        self.resource_space = resource_space
        self.action_space = list(resource_space) + ContentActions.meta_actions()

    def select_action(self, state_vec, mask=None):
        # Masked random action selection (placeholder for RL model)
        available = self.action_space if mask is None else [a for a, m in zip(self.action_space, mask) if m]
        return available[np.random.randint(len(available))] if available else None
