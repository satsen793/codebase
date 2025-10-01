import csv
import random
import os

# Constants for student generation
NUM_STUDENTS = 10000
LATENT_PARAMS = ['theta', 'alpha', 'phi', 's', 'g', 'tau', 'h']

STUDENTS_CSV = os.path.join(os.path.dirname(__file__), '../../csv/students.csv')

os.makedirs(os.path.dirname(STUDENTS_CSV), exist_ok=True)

def sample_latent_params():
    # Example: sample from uniform or beta distributions for each parameter
    return [
        round(random.betavariate(5, 3), 3),   # theta
        round(random.betavariate(2, 5), 3),   # alpha
        round(random.betavariate(2, 8), 3),   # phi
        round(random.uniform(0, 0.2), 3),     # s (slip)
        round(random.uniform(0, 0.2), 3),     # g (guess)
        round(random.uniform(0.5, 2.0), 3),   # tau (time/fatigue)
        round(random.uniform(0, 1), 3)        # h (hint usage)
    ]

def generate_students():
    rows = []
    for student_id in range(1, NUM_STUDENTS + 1):
        latents = sample_latent_params()
        rows.append([student_id] + latents)
    return rows

def write_csv(rows, path):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['student_id'] + LATENT_PARAMS)
        writer.writerows(rows)

if __name__ == '__main__':
    data = generate_students()
    write_csv(data, STUDENTS_CSV)
    print(f"Students CSV generated at {STUDENTS_CSV} with {len(data)} rows.")
