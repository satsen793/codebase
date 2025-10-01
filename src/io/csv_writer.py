import csv
import os
from datetime import datetime, timedelta
import random

# Simulate student interaction logs (e.g., for RL training)
NUM_STUDENTS = 10000
NUM_INTERACTIONS = 100000  # Large log

INTERACTIONS_CSV = os.path.join(os.path.dirname(__file__), '../../csv/student_interactions.csv')

os.makedirs(os.path.dirname(INTERACTIONS_CSV), exist_ok=True)

LO_IDS = [f"T{t}_LO{lo}" for t in range(1, 201) for lo in range(1, 11)]
ACTIONS = ['stay', 'backtrack', 'revisit', 'goto', 'group_work', 'NULL']
RESULTS = [0, 1]

start_time = datetime(2025, 1, 1)

def generate_interactions():
    rows = []
    for _ in range(NUM_INTERACTIONS):
        student_id = random.randint(1, NUM_STUDENTS)
        lo_id = random.choice(LO_IDS)
        action = random.choice(ACTIONS)
        result = random.choice(RESULTS)
        timestamp = start_time + timedelta(minutes=random.randint(0, 365*24*60))
        rows.append([
            student_id, lo_id, action, result, timestamp.strftime('%Y-%m-%d %H:%M')
        ])
    return rows

def write_csv(rows, path):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['student_id', 'lo_id', 'action', 'result', 'timestamp'])
        writer.writerows(rows)

if __name__ == '__main__':
    data = generate_interactions()
    write_csv(data, INTERACTIONS_CSV)
    print(f"Student interactions CSV generated at {INTERACTIONS_CSV} with {len(data)} rows.")
