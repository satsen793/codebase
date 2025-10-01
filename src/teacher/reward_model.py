# reward_model.py
# Computes total reward: R_total = R_student + alpha*R_jump + beta*R_content

class RewardModel:
    def __init__(self, alpha=1.0, beta=1.0):
        self.alpha = alpha
        self.beta = beta

    def r_student(self, correct, mastery_gain, time_penalty):
        # Example: +1 for correct, -1 for wrong, +mastery gain, -time penalty
        return (1 if correct else -1) + mastery_gain - time_penalty

    def r_jump(self, safe_jump, revisit, coverage, overconfident):
        # Example: +1 for safe jump, +1 for revisit, +coverage, -overconfident
        return int(safe_jump) + int(revisit) + coverage - int(overconfident)

    def r_content(self, appropriate_difficulty, diversity, scaffolding):
        # Example: +1 for appropriate, +diversity, +scaffolding
        return int(appropriate_difficulty) + diversity + scaffolding

    def total_reward(self, r_student, r_jump, r_content):
        return r_student + self.alpha * r_jump + self.beta * r_content
