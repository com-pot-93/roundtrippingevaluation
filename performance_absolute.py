import os
import csv
import json
from collections import defaultdict

RESULTS_DIR = './results'
ITER_RESULTS_DIR = './iter_results'

METRIC_LABELS = {
    't2t_eval_1': 'SMST',
    't2t_eval_2': 'SQST',
    'm2m_eval_1': 'EBSM',
    'm2m_eval_2': 'FBSM',
}
METRIC_ORDER = ['t2t_eval_1', 't2t_eval_2', 'm2m_eval_1', 'm2m_eval_2']


def parse_results_stem(stem):
    parts = stem.split('_')
    if len(parts) < 3:
        return None, None, None
    return parts[0], '_'.join(parts[1:-1]), parts[-1]


def core_name(filename):
    return os.path.splitext(filename)[0]


def get_files_to_exclude(dir_path, direction):
    """Returns the unique list of core filenames with 0 or 1 iterations for the given direction,
    based on every iter_results report json in dir_path matching that direction."""
    files_to_exclude = []

    for filename in os.listdir(dir_path):
        if not filename.endswith('_{}.json'.format(direction)):
            continue

        file_path = os.path.join(dir_path, filename)
        with open(file_path, "r") as infile:
            data = json.load(infile)

        for file_name, iterations in data.items():
            name = core_name(file_name)
            if len(iterations) <= 1 and name not in files_to_exclude:
                files_to_exclude.append(name)

    return files_to_exclude


def load_average_similarity_by_group(results_dir=RESULTS_DIR, iter_results_dir=ITER_RESULTS_DIR):
    tables = defaultdict(lambda: defaultdict(dict))

    exclude_by_direction = {
        direction: set(get_files_to_exclude(iter_results_dir, direction))
        for direction in ('m2m', 't2t')
    }

    for filename in os.listdir(results_dir):
        if not filename.endswith('.csv'):
            continue

        stem = filename[:-len('.csv')]
        llm, example, direction = parse_results_stem(stem)
        if llm is None:
            continue

        exclude_files = exclude_by_direction.get(direction, set())

        sums = defaultdict(float)
        counts = defaultdict(int)
        with open(os.path.join(results_dir, filename), 'r', newline='') as csvfile:
            for row in csv.DictReader(csvfile):
                if core_name(row['model_name']) in exclude_files:
                    continue

                for metric in METRIC_ORDER:
                    value = row.get(metric, 'N/A')
                    if value in ('', 'N/A'):
                        continue
                    sums[metric] += float(value)
                    counts[metric] += 1

        for metric in METRIC_ORDER:
            if counts[metric]:
                tables[(llm, direction)][example][metric] = sums[metric] / counts[metric]

    return tables


def print_table(label, table):
    columns = [m for m in METRIC_ORDER if any(m in row for row in table.values())]
    headers = ['Dataset'] + [METRIC_LABELS[m] for m in columns]
    rows = [[example] + ['{:.2f}'.format(table[example][m]) if m in table[example] else ''
                          for m in columns]
            for example in sorted(table)]

    widths = [max(len(str(cell)) for cell in col) for col in zip(headers, *rows)]

    print(label)
    print(' '.join(h.ljust(w) for h, w in zip(headers, widths)))
    for row in rows:
        print(' '.join(str(cell).ljust(w) for cell, w in zip(row, widths)))
    print()


def print_average_similarity_tables(results_dir=RESULTS_DIR):
    """One average-similarity table per (llm, direction) found in results/: rows = datasets,
    columns = SMST/SQST/EBSM/FBSM."""
    tables = load_average_similarity_by_group(results_dir)

    for (llm, direction), table in sorted(tables.items()):
        print_table('Table: {} pipeline, {}'.format(direction.upper(), llm), table)

if __name__ == "__main__":
    print_average_similarity_tables()
