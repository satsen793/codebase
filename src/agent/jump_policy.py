# jump_policy.py
# Policy for curriculum jumps (π_jump)
import numpy as np

# Enumerate jump actions as per proj1oct.txt
class JumpActions:
    STAY = 'stay'
    BACKTRACK = 'backtrack'
    REVISIT = 'revisit'  # revisit(ℓ) -- needs LO argument
    GOTO = 'goto'        # goto(ℓ) -- needs LO argument
    GROUP_WORK = 'group_work'  # group_work(ℓ) -- needs LO argument
    NULL = 'NULL'

    @staticmethod
    def all_actions():
        # For actions that require LO argument, you will generate them dynamically per LO
        return [JumpActions.STAY, JumpActions.BACKTRACK, JumpActions.REVISIT, JumpActions.GOTO, JumpActions.GROUP_WORK, JumpActions.NULL]

class JumpPolicy:
    def __init__(self, lo_list):
        # lo_list: list of all LO ids
        self.lo_list = lo_list
        # Build full action space: basic actions + actions per LO
        self.action_space = [JumpActions.STAY, JumpActions.BACKTRACK, JumpActions.NULL]
        self.action_space += [(JumpActions.REVISIT, lo) for lo in lo_list]
        self.action_space += [(JumpActions.GOTO, lo) for lo in lo_list]
        self.action_space += [(JumpActions.GROUP_WORK, lo) for lo in lo_list]

    def select_action(self, state_vec, mask=None):
        # Masked random action selection (placeholder for RL model)
        available = self.action_space if mask is None else [a for a, m in zip(self.action_space, mask) if m]
        return available[np.random.randint(len(available))] if available else None
