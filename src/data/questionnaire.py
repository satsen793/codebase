import csv
import random
import os

# Constants for questionnaire generation
NUM_TOPICS = 200
NUM_LOS = 10
NUM_DIFFICULTIES = 4
NUM_QUESTIONS_PER_DIFFICULTY = 8

MODALITIES = ['problem', 'video', 'reading', 'simulation']
TYPES = ['diagnostic', 'practice', 'review']

QUESTIONNAIRE_CSV = os.path.join(os.path.dirname(__file__), '../../csv/questionnaire.csv')

os.makedirs(os.path.dirname(QUESTIONNAIRE_CSV), exist_ok=True)

def generate_questionnaire():
    rows = []
    for topic in range(1, NUM_TOPICS + 1):
        for lo in range(1, NUM_LOS + 1):
            for diff in range(1, NUM_DIFFICULTIES + 1):
                for q in range(1, NUM_QUESTIONS_PER_DIFFICULTY + 1):
                    question_id = f"T{topic}_LO{lo}_D{diff}_Q{q}"
                    question_text = f"Solve the math word problem for topic {topic}, LO {lo}, difficulty {diff}, Q{q}."
                    irt_difficulty = round(random.uniform(-2, 2), 2)
                    modality = random.choice(MODALITIES)
                    qtype = random.choice(TYPES)
                    rows.append([
                        topic, lo, diff, question_id, question_text, irt_difficulty, modality, qtype
                    ])
    return rows

def write_csv(rows, path):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'topic', 'learning_outcome', 'difficulty', 'question_id', 'question_text',
            'irt_difficulty', 'modality', 'type'
        ])
        writer.writerows(rows)

if __name__ == '__main__':
    data = generate_questionnaire()
    write_csv(data, QUESTIONNAIRE_CSV)
    print(f"Questionnaire CSV generated at {QUESTIONNAIRE_CSV} with {len(data)} rows.")
