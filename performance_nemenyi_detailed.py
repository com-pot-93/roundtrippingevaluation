"""
Full, non-aggregated Nemenyi results: one row per (dataset, direction, metric, llm pair),
instead of NEMENYI_POSTHOC.py's aggregated summary.

Motivation: pairwise significance testing alone is magnitude-blind -- a p<0.05 "win" driven by a
0.01 score gap looks identical to a p<0.05 "win" driven by a 0.50 score gap once you only look at
significance. "This methodology introduces a significant oversimplification that we could call
magnitude-blindness, treating a marginal victory of 0.01% as equivalent to a clear dominance of
50%." This file reports the raw score difference alongside the Nemenyi p-value so a "win" can be
judged as statistically significant AND practically meaningful, or dismissed as significant-but-
negligible.

Unlike NEMENYI_POSTHOC.py, rows are NOT filtered to only significant Friedman tests -- this is the
full, ungated result set; the Friedman p-value/significance is reported per row for context instead.
"""

import os
import csv
import json
from itertools import combinations
from collections import defaultdict

import pandas as pd
from scipy.stats import friedmanchisquare
import scikit_posthocs as sp

RESULTS_DIR = '../results'
ITER_RESULTS_DIR = '../iter_results'

# SMST=t2t_eval_1 (sts_bert), SQST=t2t_eval_2 (text_similarity_alternative),
# EBSM=m2m_eval_1 (calculate_similarity union-weighted), FBSM=m2m_eval_2 (binarized-weight)
METRIC_LABELS = {
    't2t_eval_1': 'SMST',
    't2t_eval_2': 'SQST',
    'm2m_eval_1': 'EBSM',
    'm2m_eval_2': 'FBSM',
}
METRIC_ORDER = ['t2t_eval_1', 't2t_eval_2', 'm2m_eval_1', 'm2m_eval_2']

MIN_LLMS = 3
MIN_FILES = 3
ALPHA = 0.05

# Magnitude bands on the raw (0-1 scale) score difference between two llms' means. These are the
# thing that lets a "win" be judged meaningful or not, independent of whether it's significant.
MAGNITUDE_THRESHOLDS = [
    (0.01, "negligible"),
    (0.05, "small"),
    (0.15, "moderate"),
]
LARGE = "large"


def magnitude_for(diff):
    for threshold, label in MAGNITUDE_THRESHOLDS:
        if diff < threshold:
            return label
    return LARGE


def parse_results_stem(stem):
    """"gpt_domain_m2m" -> ("gpt", "domain", "m2m")."""
    parts = stem.split('_')
    if len(parts) < 3:
        return None, None, None
    return parts[0], '_'.join(parts[1:-1]), parts[-1]


def core_name(filename):
    """Strips the extension so m2m's .json-suffixed and t2t's .txt-suffixed keys compare equal."""
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


def load_per_file_scores(results_dir=RESULTS_DIR, iter_results_dir=ITER_RESULTS_DIR):
    """{(direction, example): {metric: {llm: {file: score}}}}, skipping excluded files (same
    exclusion principle as report_to_series.py's stability tables) and N/A scores."""
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


def build_aligned_frame(scores_by_llm):
    """scores_by_llm: {llm: {file: score}}. Returns a (files x llms) DataFrame on the files common
    to every llm present, or None if there aren't enough llms/files to compare."""
    llms = sorted(scores_by_llm)
    if len(llms) < MIN_LLMS:
        return None

    common_files = set.intersection(*(set(scores_by_llm[llm]) for llm in llms))
    if len(common_files) < MIN_FILES:
        return None

    files_sorted = sorted(common_files)
    return pd.DataFrame({llm: [scores_by_llm[llm][f] for f in files_sorted] for llm in llms})


def format_p(p):
    return '<0.0001' if p < 0.0001 else '{:.4f}'.format(p)


def compute_full_results(results_dir=RESULTS_DIR, iter_results_dir=ITER_RESULTS_DIR):
    """One row per (direction, dataset, metric, llm pair): mean score for each llm, the raw
    difference, the Nemenyi p-value, and a magnitude-aware "meaningful" verdict."""
    scores = load_per_file_scores(results_dir, iter_results_dir)

    rows = []
    for direction, example in sorted(scores):
        for metric in METRIC_ORDER:
            df = build_aligned_frame(scores[(direction, example)][metric])
            if df is None:
                continue

            friedman_stat, friedman_p = friedmanchisquare(*[df[col] for col in df.columns])
            pairwise_p = sp.posthoc_nemenyi_friedman(df)
            means = df.mean()

            for llm_x, llm_y in combinations(sorted(df.columns), 2):
                winner, loser = (llm_x, llm_y) if means[llm_x] >= means[llm_y] else (llm_y, llm_x)
                diff = means[winner] - means[loser]
                p = pairwise_p.loc[llm_x, llm_y]
                significant = p < ALPHA
                magnitude = magnitude_for(diff)
                meaningful = significant and magnitude != "negligible"

                rows.append({
                    'dataset': example,
                    'direction': direction,
                    'metric': METRIC_LABELS[metric],
                    'n': len(df),
                    'winner': winner,
                    'loser': loser,
                    'mean_winner': means[winner],
                    'mean_loser': means[loser],
                    'diff': diff,
                    'nemenyi_p': p,
                    'significant': significant,
                    'magnitude': magnitude,
                    'meaningful': meaningful,
                    'friedman_p': friedman_p,
                    'friedman_significant': friedman_p < ALPHA,
                })

    return rows


def print_table(label, headers, rows):
    widths = [max(len(str(cell)) for cell in col) for col in zip(headers, *rows)]

    print(label)
    print(' '.join(h.ljust(w) for h, w in zip(headers, widths)))
    for row in rows:
        print(' '.join(str(cell).ljust(w) for cell, w in zip(row, widths)))
    print()


def print_full_results(results_dir=RESULTS_DIR, iter_results_dir=ITER_RESULTS_DIR):
    rows = compute_full_results(results_dir, iter_results_dir)
    if not rows:
        print('No (dataset, direction, metric) group had enough llms/files to compare.')
        return

    directions = sorted({r['direction'] for r in rows})
    for direction in directions:
        table_rows = [
            [r['dataset'], r['metric'], r['n'], r['winner'], r['loser'],
             '{:.3f}'.format(r['mean_winner']), '{:.3f}'.format(r['mean_loser']),
             '{:.3f}'.format(r['diff']), format_p(r['nemenyi_p']),
             '*' if r['significant'] else '', r['magnitude'],
             'YES' if r['meaningful'] else 'no',
             format_p(r['friedman_p']), '*' if r['friedman_significant'] else '']
            for r in rows if r['direction'] == direction
        ]
        print_table(
            'Full Nemenyi results: {} pipeline'.format(direction.upper()),
            ['Dataset', 'Metric', 'n', 'Winner', 'Loser', 'MeanW', 'MeanL', 'Diff',
             'Nemenyi p', 'Sig?', 'Magnitude', 'Meaningful?', 'Friedman p', 'Sig?'],
            table_rows,
        )


if __name__ == "__main__":
    print_full_results()
