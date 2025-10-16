import csv
import os
import random

                                                 
NUM_TOPICS = 200
NUM_LOS = 10

CURRICULUM_CSV = os.path.join(os.path.dirname(__file__), '../../csv/curriculum.csv')

os.makedirs(os.path.dirname(CURRICULUM_CSV), exist_ok=True)

def generate_curriculum():
    rows = []
    for topic in range(1, NUM_TOPICS + 1):
        for lo in range(1, NUM_LOS + 1):
            lo_id = f"T{topic}_LO{lo}"
                                                                                 
            if lo == 1:
                prereq = ''
            else:
                prereq = f"T{topic}_LO{lo-1}"
                                                                       
            if lo > 1 and random.random() < 0.1 and topic > 1:
                cross_topic = f"T{topic-1}_LO{random.randint(1,NUM_LOS)}"
                prereq += f";{cross_topic}" if prereq else cross_topic
            rows.append([lo_id, topic, lo, prereq])
    return rows

def write_csv(rows, path):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['lo_id', 'topic', 'lo_number', 'prerequisites'])
        writer.writerows(rows)

if __name__ == '__main__':
    data = generate_curriculum()
    write_csv(data, CURRICULUM_CSV)
    print(f"Curriculum CSV generated at {CURRICULUM_CSV} with {len(data)} rows.")
