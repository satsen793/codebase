                      
                                                    
import numpy as np

class StudentSimulator:
    def __init__(self, latents):
        self.theta, self.alpha, self.phi, self.s, self.g, self.tau, self.h = latents
        self.mastery = {}                  

    def get_mastery(self, lo_id):
        return self.mastery.get(lo_id, 0.0)

    def update_mastery(self, lo_id, correct, difficulty):
                                          
        m = self.get_mastery(lo_id)
        if correct:
            m_new = m + self.alpha * (1 - m) * (1 / (1 + np.exp(-difficulty + self.theta)))
        else:
            m_new = m - 0.1 * m
        self.mastery[lo_id] = np.clip(m_new, 0, 1)
        return self.mastery[lo_id]

    def slip_or_guess(self, p_know):
                                                                                     
        if np.random.rand() < self.s:
            return False        
        if np.random.rand() < self.g:
            return True          
        return np.random.rand() < p_know
