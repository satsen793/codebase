import csv
import os

# Four environments, each with different latent parameter priors
ENVIRONMENTS = [
    {
        'env_id': 'SchoolA',
        'theta_alpha': 'Beta(5,3)',
        'phi': 'Beta(2,8)',
        's': 'Uniform(0,0.1)',
        'g': 'Uniform(0,0.1)',
        'tau': 'Uniform(0.5,1.0)',
        'h': 'Uniform(0,0.5)'
    },
    {
        'env_id': 'SchoolB',
        'theta_alpha': 'Beta(2,5)',
        'phi': 'Beta(3,7)',
        's': 'Uniform(0,0.2)',
        'g': 'Uniform(0,0.2)',
        'tau': 'Uniform(1.0,1.5)',
        'h': 'Uniform(0.2,0.7)'
    },
    {
        'env_id': 'SchoolC',
        'theta_alpha': 'Beta(6,2)',
        'phi': 'Beta(1,5)',
        's': 'Uniform(0,0.15)',
        'g': 'Uniform(0,0.15)',
        'tau': 'Uniform(0.7,1.2)',
        'h': 'Uniform(0.1,0.6)'
    },
    {
        'env_id': 'SchoolD',
        'theta_alpha': 'Beta(4,4)',
        'phi': 'Beta(2,6)',
        's': 'Uniform(0,0.05)',
        'g': 'Uniform(0,0.05)',
        'tau': 'Uniform(1.2,2.0)',
        'h': 'Uniform(0.3,0.9)'
    }
]

ENVIRONMENTS_CSV = os.path.join(os.path.dirname(__file__), '../../csv/environments.csv')

os.makedirs(os.path.dirname(ENVIRONMENTS_CSV), exist_ok=True)

def write_csv(rows, path):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['env_id', 'theta_alpha', 'phi', 's', 'g', 'tau', 'h'])
        for row in rows:
            writer.writerow([
                row['env_id'], row['theta_alpha'], row['phi'], row['s'], row['g'], row['tau'], row['h']
            ])

if __name__ == '__main__':
    write_csv(ENVIRONMENTS, ENVIRONMENTS_CSV)
    print(f"Environments CSV generated at {ENVIRONMENTS_CSV} with {len(ENVIRONMENTS)} rows.")
