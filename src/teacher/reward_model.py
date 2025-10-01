# reward_model.py
# Computes total reward: R_total = R_student + alpha*R_jump + beta*R_content

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
        # +1 for correct (scaled by slip/guess), -1 for wrong, +mastery gain, -time penalty
        return (1 if correct else -1) * (1 - slip) + mastery_gain - time_penalty + guess

    def r_jump(self, jump_info):
        # jump_info: dict with keys for all jump shaping rules
        r = 0
        # Prerequisite safety
        if not jump_info.get('prereq_safe', True):
            r -= 1
        # ZPD alignment (optimal challenge zone)
        if jump_info.get('zpd', False):
            r += 1
        # Coverage (new LOs visited)
        r += self.lambda_cov * jump_info.get('coverage', 0)
        # Staleness (staying too long)
        r -= self.lambda_stale * jump_info.get('stale', 0)
        # Retention trigger (spaced revisit)
        if jump_info.get('retention', False):
            r += 1
        # Overconfidence penalty
        if jump_info.get('overconfident', False):
            r -= 1
        # Velocity penalty
        r -= self.lambda_v * max(0, abs(jump_info.get('jump_vel', 0)) - jump_info.get('v_max', 1))
        return r

    def r_content(self, content_info):
        # content_info: dict with keys for all content shaping rules
        r = 0
        # Difficulty appropriateness
        if not content_info.get('appropriate_difficulty', True):
            r -= 1
        # Modality diversity
        r -= self.lambda_mod * content_info.get('modality_switch', 0)
        # Type balance (diagnostic/practice)
        r -= self.lambda_type * content_info.get('type_imbalance', 0)
        # Scaffolding awareness
        if content_info.get('scaffold', False):
            r += 1
        if content_info.get('hint_on_easy', False):
            r -= 1
        # Exposure control (overuse)
        r -= self.lambda_exp * content_info.get('overexposed', 0)
        return r

    def total_reward(self, r_student, r_jump, r_content):
        return r_student + self.alpha * r_jump + self.beta * r_content
