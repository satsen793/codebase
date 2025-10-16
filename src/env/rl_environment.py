                   
                                                                      
import numpy as np
from env.student_simulator import StudentSimulator
from env.curriculum_dag import CurriculumDAG
from agent.jump_policy import JumpActions
from agent.content_policy import ContentActions

class RLEnvironment:
    def __init__(self, student_latents, curriculum_csv, resource_map, question_bank):
        self.curriculum = CurriculumDAG(curriculum_csv)
        self.resource_map = resource_map                                       
        self.question_bank = question_bank                                      
        self.student_latents = student_latents                              
        self.reset()

    def reset(self):
                              
        latents = self.student_latents[np.random.randint(len(self.student_latents))]
        self.student = StudentSimulator(latents)
        self.current_lo = list(self.resource_map.keys())[0]                     
        self.done = False
        self.t = 0
        self.history = []
        return self._get_state()

    def _get_state(self):
                                                               
        return {
            'mastery': self.student.mastery,
            'current_lo': self.current_lo,
            't': self.t,
            'latents': [self.student.theta, self.student.alpha, self.student.phi, self.student.s, self.student.g, self.student.tau, self.student.h],
            'history': self.history
        }

    def step(self, jump_action, content_action):
                            
        if isinstance(jump_action, tuple) and jump_action[0] in [JumpActions.GOTO, JumpActions.REVISIT, JumpActions.GROUP_WORK]:
            target_lo = jump_action[1]
            self.current_lo = target_lo
        elif jump_action == JumpActions.BACKTRACK:
            prereqs = self.curriculum.lo_prereqs.get(self.current_lo, [])
            if prereqs:
                self.current_lo = prereqs[0]
        elif jump_action == JumpActions.NULL:
            self.done = True
            return self._get_state(), 0, self.done, {'info': 'Session ended'}
                                                   
        correct = np.random.rand() < 0.7                            
        difficulty = self.question_bank.get(content_action, {}).get('irt_difficulty', 0)
        mastery_gain = self.student.update_mastery(self.current_lo, correct, difficulty)
        self.t += 1
        self.history.append((jump_action, content_action, correct))
                                        
        reward = 1 if correct else -1
        if self.t > 50:
            self.done = True
        return self._get_state(), reward, self.done, {'correct': correct, 'mastery_gain': mastery_gain}

    def available_jump_actions(self):
                                                                                           
        unlocked = self.curriculum.unlocked(self.student.mastery)
        actions = [JumpActions.STAY, JumpActions.BACKTRACK, JumpActions.NULL]
        actions += [(JumpActions.GOTO, lo) for lo in unlocked if lo != self.current_lo]
        actions += [(JumpActions.REVISIT, lo) for lo in self.student.mastery if self.student.mastery[lo] > 0.7 and lo != self.current_lo]
        actions += [(JumpActions.GROUP_WORK, lo) for lo in unlocked if lo != self.current_lo]
        return actions

    def available_content_actions(self):
                                                                                           
        resources = self.resource_map.get(self.current_lo, [])
        return list(resources) + ContentActions.meta_actions()
