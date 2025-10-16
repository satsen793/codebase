                
                                      
import numpy as np

                                            
class JumpActions:
    STAY = 'stay'
    BACKTRACK = 'backtrack'
    REVISIT = 'revisit'                                   
    GOTO = 'goto'                                      
    GROUP_WORK = 'group_work'                                      
    NULL = 'NULL'

    @staticmethod
    def all_actions():
                                                                                         
        return [JumpActions.STAY, JumpActions.BACKTRACK, JumpActions.REVISIT, JumpActions.GOTO, JumpActions.GROUP_WORK, JumpActions.NULL]

class JumpPolicy:
    def __init__(self, lo_list):
                                     
        self.lo_list = lo_list
                                                                 
        self.action_space = [JumpActions.STAY, JumpActions.BACKTRACK, JumpActions.NULL]
        self.action_space += [(JumpActions.REVISIT, lo) for lo in lo_list]
        self.action_space += [(JumpActions.GOTO, lo) for lo in lo_list]
        self.action_space += [(JumpActions.GROUP_WORK, lo) for lo in lo_list]

    def select_action(self, state_vec, mask=None):
                                                                   
        available = self.action_space if mask is None else [a for a, m in zip(self.action_space, mask) if m]
        return available[np.random.randint(len(available))] if available else None
