# curriculum_dag.py
# Handles DAG unlock logic for curriculum
import networkx as nx

class CurriculumDAG:
    def __init__(self, curriculum_csv):
        self.graph = nx.DiGraph()
        self.lo_prereqs = {}
        self.load_from_csv(curriculum_csv)

    def load_from_csv(self, path):
        import csv
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                lo_id = row['lo_id']
                prereqs = row['prerequisites'].split(';') if row['prerequisites'] else []
                self.graph.add_node(lo_id)
                for p in prereqs:
                    self.graph.add_edge(p, lo_id)
                self.lo_prereqs[lo_id] = prereqs

    def unlocked(self, mastery_dict, threshold=0.8):
        # Returns set of unlocked LO ids given current mastery
        unlocked = set()
        for lo, prereqs in self.lo_prereqs.items():
            if all(mastery_dict.get(p, 0) >= threshold for p in prereqs):
                unlocked.add(lo)
        return unlocked

    def is_unlocked(self, lo_id, mastery_dict, threshold=0.8):
        return all(mastery_dict.get(p, 0) >= threshold for p in self.lo_prereqs.get(lo_id, []))
