import os
import csv
import json
from collections import defaultdict

import pandas as pd
from scipy.stats import friedmanchisquare

RESULTS_DIR = './results'
ITER_RESULTS_DIR = './iter_results'

METRIC_LABELS = {
    't2t_eval_1': 'SMST',
    't2t_eval_2': 'SQST',
    'm2m_eval_1': 'EBSM',
    'm2m_eval_2': 'FBSM',
}
METRIC_ORDER = ['t2t_eval_1', 't2t_eval_2', 'm2m_eval_1', 'm2m_eval_2']

MIN_LLMS = 3
MIN_FILES = 3


def parse_results_stem(stem):
    parts = stem.split('_')
    if len(parts) < 3:
        return None, None, None
    return parts[0], '_'.join(parts[1:-1]), parts[-1]


def core_name(filename):
    return os.path.splitext(filename)[0]


def get_files_to_exclude(dir_path, direction):
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


def load_per_file_scores(results_dir=RESULTS_DIR, iter_results_dir=ITER_RESULTS_DIR):
    exclude_by_direction = {
        direction: set(get_files_to_exclude(iter_results_dir, direction))
        for direction in ('m2m', 't2t')
    }

    scores = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    for filename in os.listdir(results_dir):
        if not filename.endswith('.csv'):
            continue

        stem = filename[:-len('.csv')]
        llm, example, direction = parse_results_stem(stem)
        if llm is None:
            continue

        exclude_files = exclude_by_direction.get(direction, set())

        with open(os.path.join(results_dir, filename), 'r', newline='') as csvfile:
            for row in csv.DictReader(csvfile):
                file_name = core_name(row['model_name'])
                if file_name in exclude_files:
                    continue

                for metric in METRIC_ORDER:
                    value = row.get(metric, 'N/A')
                    if value in ('', 'N/A'):
                        continue
                    scores[(direction, example)][metric][llm][file_name] = float(value)

    return scores


def run_friedman(scores_by_llm):
    llms = sorted(scores_by_llm)
    if len(llms) < MIN_LLMS:
        return None

    common_files = set.intersection(*(set(scores_by_llm[llm]) for llm in llms))
    if len(common_files) < MIN_FILES:
        return None

    files_sorted = sorted(common_files)
    df = pd.DataFrame({llm: [scores_by_llm[llm][f] for f in files_sorted] for llm in llms})

    stat, p = friedmanchisquare(*[df[llm] for llm in llms])
    mean_ranks = df.rank(axis=1).mean().sort_values(ascending=False)

    return len(files_sorted), llms, stat, p, mean_ranks


def print_friedman_report(results_dir=RESULTS_DIR, iter_results_dir=ITER_RESULTS_DIR):
    scores = load_per_file_scores(results_dir, iter_results_dir)

    for direction, example in sorted(scores):
        print('Friedman test: {} pipeline, {}'.format(direction.upper(), example))

        for metric in METRIC_ORDER:
            label = METRIC_LABELS[metric]
            result = run_friedman(scores[(direction, example)][metric])

            if result is None:
                print('  {}: not enough matching llms/files to run the test'.format(label))
                continue

            n, llms, stat, p, mean_ranks = result
            print('  {} (n={}, llms={}): chi2={:.4f}, p={:.4f}'.format(label, n, ', '.join(llms), stat, p))
            for llm, rank in mean_ranks.items():
                print('    {}: mean rank = {:.2f}'.format(llm, rank))

        print()


if __name__ == "__main__":
    print_friedman_report()
