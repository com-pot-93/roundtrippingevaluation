import os
import csv
import json
from collections import defaultdict, Counter

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


def format_p(p):
    return '<0.0001' if p < 0.0001 else '{:.4f}'.format(p)


def print_table(label, headers, rows):
    widths = [max(len(str(cell)) for cell in col) for col in zip(headers, *rows)]

    print(label)
    print(' '.join(h.ljust(w) for h, w in zip(headers, widths)))
    for row in rows:
        print(' '.join(str(cell).ljust(w) for cell, w in zip(row, widths)))
    print()


def print_summary_table(all_results):
    llm_columns = sorted({llm for _, _, _, llms in all_results for llm in llms})

    rows = []
    for metric in METRIC_ORDER:
        metric_results = [r for r in all_results if r[0] == metric]
        if not metric_results:
            continue

        wins = Counter(winner for _, winner, _, _ in metric_results)
        total = len(metric_results)
        significant = sum(1 for _, _, sig, _ in metric_results if sig == '*')
        sig_rate = '{:.0f}%'.format(100 * significant / total)

        rows.append(
            [METRIC_LABELS[metric]] + [wins.get(llm, 0) for llm in llm_columns]
            + ['{} / {}'.format(significant, total), sig_rate]
        )

    if rows:
        headers = ['Metric'] + ['{} wins'.format(llm) for llm in llm_columns] + ['Significant / Total', 'Sig. Rate']
        print_table('Summary across all datasets and directions', headers, rows)


def print_friedman_report(results_dir=RESULTS_DIR, iter_results_dir=ITER_RESULTS_DIR):
    scores = load_per_file_scores(results_dir, iter_results_dir)

    all_results = []
    directions = sorted({direction for direction, _ in scores})
    for direction in directions:
        rows = []
        for _, example in sorted(k for k in scores if k[0] == direction):
            for metric in METRIC_ORDER:
                result = run_friedman(scores[(direction, example)][metric])
                if result is None:
                    continue

                n, llms, stat, p, mean_ranks = result
                winner = mean_ranks.index[0]
                sig = '*' if p < 0.05 else ''
                rows.append([example, METRIC_LABELS[metric], '{:.2f}'.format(stat), format_p(p), winner, sig])
                all_results.append((metric, winner, sig, llms))

        if rows:
            print_table(
                'Friedman test: {} pipeline'.format(direction.upper()),
                ['Dataset', 'Metric', 'chi2', 'p', 'Winner', 'Sig?'],
                rows,
            )

    print_summary_table(all_results)


if __name__ == "__main__":
    print_friedman_report()
