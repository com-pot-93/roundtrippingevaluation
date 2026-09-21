import os
import csv
import json

import numpy as np
from scipy import stats

RESULTS_DIR = '../results'
ITER_RESULTS_DIR = '../iter_results'

MIN_FILES = 4  # need n > 3 for the Fisher z-transform CI


def parse_results_stem(stem):
    parts = stem.split('_')
    if len(parts) < 3:
        return None, None, None
    return parts[0], '_'.join(parts[1:-1]), parts[-1]


def core_name(filename):
    """Strips the extension so m2m's .json-suffixed and t2t's .txt-suffixed keys compare equal."""
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


def spearman_with_ci(x, y):
    n = len(x)
    rho, p = stats.spearmanr(x, y)

    z = np.arctanh(rho)
    se = 1 / np.sqrt(n - 3)
    ci_lower, ci_upper = np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)

    return rho, p, ci_lower, ci_upper


def compute_correlations(results_dir=RESULTS_DIR, iter_results_dir=ITER_RESULTS_DIR):
    exclude_by_direction = {
        direction: set(get_files_to_exclude(iter_results_dir, direction))
        for direction in ('m2m', 't2t')
    }

    rows = []
    for filename in sorted(os.listdir(results_dir)):
        if not filename.endswith('.csv'):
            continue

        stem = filename[:-len('.csv')]
        llm, example, direction = parse_results_stem(stem)
        if llm is None:
            continue

        exclude_files = exclude_by_direction.get(direction, set())

        text_eval_1, model_eval_2 = [], []
        with open(os.path.join(results_dir, filename), 'r', newline='') as csvfile:
            for row in csv.DictReader(csvfile):
                if core_name(row['model_name']) in exclude_files:
                    continue

                t2t_1 = row.get('t2t_eval_1', 'N/A')
                m2m_2 = row.get('t2t_eval_2', 'N/A')
                #t2t_1 = row.get('m2m_eval_1', 'N/A')
                #m2m_2 = row.get('m2m_eval_2', 'N/A')
                if t2t_1 in ('', 'N/A') or m2m_2 in ('', 'N/A'):
                    continue

                text_eval_1.append(float(t2t_1))
                model_eval_2.append(float(m2m_2))

        n = len(text_eval_1)
        if n < MIN_FILES:
            continue

        rho, p, ci_lower, ci_upper = spearman_with_ci(text_eval_1, model_eval_2)
        rows.append({
            'llm': llm,
            'dataset': example,
            'direction': direction,
            'n': n,
            'rho': rho,
            'p_value': p,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
        })

    return rows


def format_p(p):
    """Tiny p-values round to 0.0000 at 4 decimals, so switch to scientific notation below 0.0001."""
    return '{:.2e}'.format(p) if p < 0.0001 else '{:.4f}'.format(p)


def print_pivot_correlation_table(rows):
    if not rows:
        print('No results csv had enough non-excluded files (n >= {}) to compute a correlation.'.format(MIN_FILES))
        return

    directions = sorted({r['direction'] for r in rows})
    llms = sorted({r['llm'] for r in rows})
    datasets = sorted({r['dataset'] for r in rows})
    lookup = {(r['dataset'], r['direction'], r['llm']): r for r in rows}
    col_specs = [(direction, llm) for direction in directions for llm in llms]

    def cell(dataset, direction, llm, key, fmt):
        r = lookup.get((dataset, direction, llm))
        return fmt(r[key]) if r else ''

    data_rows = []
    for dataset in datasets:
        row = [dataset]
        for direction, llm in col_specs:
            row.append(cell(dataset, direction, llm, 'rho', '{:.3f}'.format))
            row.append(cell(dataset, direction, llm, 'p_value', format_p))
        data_rows.append(row)

    sub_headers = [h for _ in col_specs for h in ('cor.', 'p-value')]
    llm_header = [h for direction, llm in col_specs for h in (llm.upper(), '')]
    direction_header = []
    for direction in directions:
        direction_header += [direction.upper()] + [''] * (2 * len(llms) - 1)

    n_cols = 1 + 2 * len(col_specs)
    widths = [max(len('Dataset'), *(len(row[0]) for row in data_rows))]
    for i in range(1, n_cols):
        candidates = [len(sub_headers[i - 1]), len(llm_header[i - 1]), len(direction_header[i - 1])]
        candidates += [len(row[i]) for row in data_rows]
        widths.append(max(candidates))

    def print_row(cells):
        print(' '.join(str(c).ljust(w) for c, w in zip(cells, widths)))

    print('Spearman correlation: text_eval_1 (SMST) vs model_eval_2 (FBSM)')
    print_row(['Pipeline'] + direction_header)
    print_row(['LLM'] + llm_header)
    print_row(['Value'] + sub_headers)
    for row in data_rows:
        print_row(row)
    print()


if __name__ == "__main__":
    print_pivot_correlation_table(compute_correlations())
