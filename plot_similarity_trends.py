import os
import csv
import json
from collections import defaultdict

import matplotlib.pyplot as plt

RESULTS_DIR = '../results'
ITER_RESULTS_DIR = '../iter_results'
PICS_DIR = './pics'

# SMST=t2t_eval_1 (sts_bert), SQST=t2t_eval_2 (text_similarity_alternative),
# EBSM=m2m_eval_1 (calculate_similarity union-weighted), FBSM=m2m_eval_2 (binarized-weight)
METRIC_LABELS = {
    't2t_eval_1': 'SMST',
    't2t_eval_2': 'SQST',
    'm2m_eval_1': 'EBSM',
    'm2m_eval_2': 'FBSM',
}
METRIC_ORDER = ['t2t_eval_1', 't2t_eval_2', 'm2m_eval_1', 'm2m_eval_2']

# Two colour families so text metrics and model metrics are visually distinct at a glance.
METRIC_COLORS = {
    't2t_eval_1': '#1a5fb4',  # dark blue
    't2t_eval_2': '#62a0ea',  # light blue
    'm2m_eval_1': '#a51d2d',  # dark red
    'm2m_eval_2': '#f66151',  # light red
}
METRIC_MARKERS = {
    't2t_eval_1': 'o',
    't2t_eval_2': 's',
    'm2m_eval_1': 'o',
    'm2m_eval_2': 's',
}

# Fixed x-axis order; any dataset found in the data but not listed here is appended afterward,
# alphabetically.
DATASET_ORDER = ['domain', 'mad', 'pet', 'sapsam', 'realset']


def order_datasets(datasets):
    present = set(datasets)
    ordered = [d for d in DATASET_ORDER if d in present]
    ordered += sorted(present - set(ordered))
    return ordered


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


def load_average_similarity_by_group(results_dir=RESULTS_DIR, iter_results_dir=ITER_RESULTS_DIR):
    """{(llm, direction): {example: {metric: average across the dataset's non-excluded files}}},
    read from results/*.csv, skipping files with 0 or 1 iterations. Identical computation to
    AVERAGE_SIMILARITY.py, duplicated standalone so the numbers here match it exactly."""
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


def plot_average_line_charts(direction, results_dir=RESULTS_DIR, iter_results_dir=ITER_RESULTS_DIR):
    """One figure per direction: subplots = llms, x-axis = dataset, y-axis = average similarity,
    one line per metric (all 4 on the same axes), text metrics in blue shades and model metrics
    in red shades so they're easy to tell apart."""
    tables = load_average_similarity_by_group(results_dir, iter_results_dir)
    llms = sorted({llm for llm, d in tables if d == direction})
    if not llms:
        print('No results found for direction={}'.format(direction))
        return

    fig, axes = plt.subplots(1, len(llms), figsize=(7 * len(llms), 6), sharey=True)
    if len(llms) == 1:
        axes = [axes]

    for ax, llm in zip(axes, llms):
        per_dataset = tables[(llm, direction)]
        datasets = order_datasets(per_dataset)

        for metric in METRIC_ORDER:
            xs = [d for d in datasets if metric in per_dataset[d]]
            ys = [per_dataset[d][metric] for d in xs]
            if not xs:
                continue
            ax.plot(xs, ys, marker=METRIC_MARKERS[metric], color=METRIC_COLORS[metric],
                     label=METRIC_LABELS[metric], linewidth=2)

        ax.set_title(llm.upper())
        ax.set_xlabel('Dataset')
        ax.tick_params(axis='x', rotation=45)
        ax.set_ylabel('Average similarity score')
        ax.legend()

    plt.tight_layout()
    os.makedirs(PICS_DIR, exist_ok=True)
    base = os.path.join(PICS_DIR, 'average_similarity_lines - {}'.format(direction))
    plt.savefig(base + '.png', dpi=300)
    plt.savefig(base + '.pdf')
    plt.savefig(base + '.svg')
    print('Saved {}.png, {}.pdf and {}.svg'.format(base, base, base))


if __name__ == "__main__":
    plot_average_line_charts('m2m')
    plot_average_line_charts('t2t')
