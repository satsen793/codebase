                 
                                                                            

import numpy as np

class RewardModel:
    def __init__(self, alpha=1.0, beta=1.0, lambda_cov=0.1, lambda_stale=0.1, lambda_mod=0.1, lambda_type=0.1, lambda_exp=0.05, lambda_v=0.1):
        self.alpha = alpha
        self.beta = beta
        self.lambda_cov = lambda_cov
        self.lambda_stale = lambda_stale
        self.lambda_mod = lambda_mod
        self.lambda_type = lambda_type
        self.lambda_exp = lambda_exp
        self.lambda_v = lambda_v

    def r_student(self, correct, mastery_gain, time_penalty, slip, guess):
                                                                                           
        return (1 if correct else -1) * (1 - slip) + mastery_gain - time_penalty + guess

    def r_jump(self, jump_info):
                                                              
        r = 0
                             
        if not jump_info.get('prereq_safe', True):
            r -= 1
                                                
        if jump_info.get('zpd', False):
            r += 1
                                    
        r += self.lambda_cov * jump_info.get('coverage', 0)
                                      
        r -= self.lambda_stale * jump_info.get('stale', 0)
                                            
        if jump_info.get('retention', False):
            r += 1
                                
        if jump_info.get('overconfident', False):
            r -= 1
                          
        r -= self.lambda_v * max(0, abs(jump_info.get('jump_vel', 0)) - jump_info.get('v_max', 1))
        return r

    def r_content(self, content_info):
                                                                    
        r = 0
                                    
        if not content_info.get('appropriate_difficulty', True):
            r -= 1
                            
        r -= self.lambda_mod * content_info.get('modality_switch', 0)
                                            
        r -= self.lambda_type * content_info.get('type_imbalance', 0)
                               
        if content_info.get('scaffold', False):
            r += 1
        if content_info.get('hint_on_easy', False):
            r -= 1
                                    
        r -= self.lambda_exp * content_info.get('overexposed', 0)
        return r

    def total_reward(self, r_student, r_jump, r_content):
        return r_student + self.alpha * r_jump + self.beta * r_content
