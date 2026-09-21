import os
import csv
import json
import math
from collections import defaultdict
from itertools import combinations

import pandas as pd
from scipy.stats import friedmanchisquare, studentized_range
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


def print_table(label, headers, rows):
    widths = [max(len(str(cell)) for cell in col) for col in zip(headers, *rows)]

    print(label)
    print(' '.join(h.ljust(w) for h, w in zip(headers, widths)))
    for row in rows:
        print(' '.join(str(cell).ljust(w) for cell, w in zip(row, widths)))
    print()


def print_nemenyi_matrix(label, pairwise_p):
    llms = list(pairwise_p.columns)
    headers = [''] + llms
    rows = []
    for llm in llms:
        row = [llm]
        for other in llms:
            if llm == other:
                row.append('-')
            else:
                p = pairwise_p.loc[llm, other]
                row.append(format_p(p) + ('*' if p < ALPHA else ''))
        rows.append(row)

    print_table(label, headers, rows)


def critical_difference(k, n, alpha=ALPHA):
    """Nemenyi critical difference: CD = q_alpha * sqrt(k*(k+1) / (6*n)), where q_alpha is the
    studentized range critical value (infinite df) divided by sqrt(2) -- this reproduces the
    standard tabulated q_alpha values (e.g. 2.343 for k=3 at alpha=.05, Demšar 2006)."""
    q = studentized_range.ppf(1 - alpha, k, float('inf')) / math.sqrt(2)
    return q * math.sqrt(k * (k + 1) / (6 * n))


def print_mean_ranks(mean_ranks, cd):
    rows = [[llm, '{:.2f}'.format(rank)] for llm, rank in mean_ranks.items()]
    print_table('Mean ranks (critical difference = {:.2f} at alpha = {})'.format(cd, ALPHA), ['LLM', 'Mean rank'], rows)


def print_rank_diffs(label, mean_ranks, cd):
    """Flat pair list (largest difference first) instead of an llm x llm grid: each row is one
    pair's absolute mean-rank difference, marked significant when it exceeds the critical
    difference (agrees with the '*' marks in the p-value matrix, since both express the same test)."""
    llms = list(mean_ranks.index)
    diffs = [
        (a, b, abs(mean_ranks[a] - mean_ranks[b]))
        for a, b in combinations(llms, 2)
    ]
    diffs.sort(key=lambda item: item[2], reverse=True)

    rows = [
        ['{} vs {}'.format(a, b), '{:.2f}'.format(diff), '*' if diff > cd else '']
        for a, b, diff in diffs
    ]
    print_table(label, ['Pair', 'Rank diff', 'Sig?'], rows)


def print_overall_summary(pair_stats, num_tests):
    total = sum(total for _, total in pair_stats.values())
    significant = sum(sig for sig, _ in pair_stats.values())
    pct = 100 * significant / total if total else 0

    print('Overall results')
    print()
    print('{} total pairwise comparisons across {} tests'.format(total, num_tests))
    print('{} were significant ({:.1f}%)'.format(significant, pct))
    print()

    def rate(item):
        sig, total = item[1]
        return sig / total if total else 0

    rows = []
    for (llm_a, llm_b), (sig, total) in sorted(pair_stats.items(), key=rate, reverse=True):
        rows.append(['{} vs {}'.format(llm_a, llm_b), '{}/{} ({:.1f}%)'.format(sig, total, 100 * sig / total if total else 0)])

    if rows:
        print_table('Pairwise comparison summary', ['Pair', 'Significant / Total'], rows)


def print_nemenyi_report(results_dir=RESULTS_DIR, iter_results_dir=ITER_RESULTS_DIR):
    """For every (direction, dataset, metric) whose Friedman test comes out significant (p < 0.05),
    runs the Nemenyi post-hoc test to see which specific llm pairs differ, and prints the pairwise
    p-value matrix. Groups with a non-significant (or unrunnable) Friedman test are skipped, since
    a post-hoc test isn't meaningful without a significant omnibus result. Finishes with an overall
    summary: total pairwise comparisons, how many were significant, and a per-llm-pair breakdown."""
    scores = load_per_file_scores(results_dir, iter_results_dir)

    pair_stats = defaultdict(lambda: [0, 0])
    num_tests = 0

    for direction, example in sorted(scores):
        for metric in METRIC_ORDER:
            df = build_aligned_frame(scores[(direction, example)][metric])
            if df is None:
                continue

            stat, p = friedmanchisquare(*[df[col] for col in df.columns])
            if p >= ALPHA:
                continue

            pairwise_p = sp.posthoc_nemenyi_friedman(df)
            n, k = df.shape
            mean_ranks = df.rank(axis=1).mean().sort_values(ascending=False)
            cd = critical_difference(k, n)

            test_label = '{} pipeline, {}, {} (Friedman p={})'.format(
                direction.upper(), example, METRIC_LABELS[metric], format_p(p)
            )
            print_nemenyi_matrix('Nemenyi p-values: ' + test_label, pairwise_p)
            print_mean_ranks(mean_ranks, cd)
            print_rank_diffs('Rank differences: ' + test_label, mean_ranks, cd)

            num_tests += 1
            for llm_a, llm_b in combinations(sorted(pairwise_p.columns), 2):
                entry = pair_stats[(llm_a, llm_b)]
                entry[1] += 1
                if pairwise_p.loc[llm_a, llm_b] < ALPHA:
                    entry[0] += 1

    print_overall_summary(pair_stats, num_tests)


if __name__ == "__main__":
    print_nemenyi_report()
