import csv
import os
import random

                                                                
NUM_TOPICS = 200
NUM_LOS = 10
NUM_DIFFICULTIES = 4
NUM_QUESTIONS_PER_DIFFICULTY = 8

REAL_CONTENT_POOL = [
    f"real_vid_{i}" for i in range(1, 1001)
] + [
    f"real_prob_{i}" for i in range(1, 2001)
] + [
    f"real_read_{i}" for i in range(1, 1001)
]

MAPPER_CSV = os.path.join(os.path.dirname(__file__), '../../csv/real_content_mapper.csv')

os.makedirs(os.path.dirname(MAPPER_CSV), exist_ok=True)

def generate_mapping():
    rows = []
    for topic in range(1, NUM_TOPICS + 1):
        for lo in range(1, NUM_LOS + 1):
            for diff in range(1, NUM_DIFFICULTIES + 1):
                for q in range(1, NUM_QUESTIONS_PER_DIFFICULTY + 1):
                    synthetic_id = f"T{topic}_LO{lo}_D{diff}_Q{q}"
                    real_id = random.choice(REAL_CONTENT_POOL)
                    rows.append([synthetic_id, real_id])
    return rows

def write_csv(rows, path):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['synthetic_question_id', 'real_content_id'])
        writer.writerows(rows)

if __name__ == '__main__':
    data = generate_mapping()
    write_csv(data, MAPPER_CSV)
    print(f"Real content mapping CSV generated at {MAPPER_CSV} with {len(data)} rows.")
