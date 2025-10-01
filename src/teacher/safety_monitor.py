# safety_monitor.py
# Monitors for 8-mistake flag and other safety checks

class SafetyMonitor:
    def __init__(self, mistake_limit=8):
        self.mistake_limit = mistake_limit
        self.mistake_counts = {}  # student_id -> count

    def record_mistake(self, student_id):
        self.mistake_counts[student_id] = self.mistake_counts.get(student_id, 0) + 1
        return self.mistake_counts[student_id]

    def should_flag(self, student_id):
        return self.mistake_counts.get(student_id, 0) >= self.mistake_limit

    def reset(self, student_id):
        self.mistake_counts[student_id] = 0
