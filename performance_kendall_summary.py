import os
import csv
import json
from collections import defaultdict

import pandas as pd
from scipy.stats import friedmanchisquare

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

# Kendall's W agreement bands, in ascending order: (upper_bound, label)
AGREEMENT_THRESHOLDS = [
    (0.1, "negligible"),
    (0.3, "weak"),
    (0.5, "moderate"),
    (0.7, "strong"),
]
VERY_STRONG = "very strong"


def agreement_for(w):
    for threshold, label in AGREEMENT_THRESHOLDS:
        if w < threshold:
            return label
    return VERY_STRONG


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


def kendalls_w(df):
    """Kendall's W (coefficient of concordance) among the llms' rankings, derived from the same
    (uncorrected-for-ties) Friedman chi-square statistic FRIEDMAN_TEST.py reports:
    W = chi2 / (n * (k - 1)), where n = subjects (files) and k = raters (llms). Ranges 0 (no
    agreement) to 1 (perfect agreement)."""
    n, k = df.shape
    stat, p = friedmanchisquare(*[df[col] for col in df.columns])
    w = stat / (n * (k - 1))
    return w, stat, p, n, k


def print_table(label, headers, rows):
    widths = [max(len(str(cell)) for cell in col) for col in zip(headers, *rows)]

    print(label)
    print(' '.join(h.ljust(w) for h, w in zip(headers, widths)))
    for row in rows:
        print(' '.join(str(cell).ljust(w) for cell, w in zip(row, widths)))
    print()


def print_kendall_report(results_dir=RESULTS_DIR, iter_results_dir=ITER_RESULTS_DIR):
    """One table per direction: rows = (dataset, metric) with a matched-llm comparison available,
    columns = n (files), k (llms), Kendall's W, and its agreement-strength label."""
    scores = load_per_file_scores(results_dir, iter_results_dir)

    directions = sorted({direction for direction, _ in scores})
    for direction in directions:
        rows = []
        for _, example in sorted(key for key in scores if key[0] == direction):
            for metric in METRIC_ORDER:
                df = build_aligned_frame(scores[(direction, example)][metric])
                if df is None:
                    continue

                w, stat, p, n, k = kendalls_w(df)
                rows.append([example, METRIC_LABELS[metric], n, k, '{:.3f}'.format(w), agreement_for(w)])

        if rows:
            print_table(
                "Kendall's W: {} pipeline".format(direction.upper()),
                ['Dataset', 'Metric', 'n', 'k', 'W', 'Agreement'],
                rows,
            )


if __name__ == "__main__":
    print_kendall_report()
