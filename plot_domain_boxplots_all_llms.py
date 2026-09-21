import os
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

RESULTS_DIR = '../results'
ITER_RESULTS_DIR = '../iter_results'
PICS_DIR = './pics'

DATASET = 'domain'
DIRECTION = 't2t'
METRIC_COLUMN = 't2t_eval_2'
METRIC_LABEL = 'Text Similarity Score (SQST)'
#DIRECTION = 'm2m'
#METRIC_COLUMN = 'm2m_eval_2'
#METRIC_LABEL = 'Model Similarity Score (FBSM)'

COLOURS = ['#ffa8db', '#f35eb5']


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


def find_llms(results_dir, dataset, direction):
    """Every llm with a results/{llm}_{dataset}_{direction}.csv file."""
    suffix = '_{}_{}.csv'.format(dataset, direction)
    return sorted(
        filename[:-len(suffix)]
        for filename in os.listdir(results_dir)
        if filename.endswith(suffix)
    )


def load_domain_scores(llm, dataset, direction, metric_column, exclude_files):
    """Loads results/{llm}_{dataset}_{direction}.csv, drops excluded/N-A rows, and groups the
    per-file scores by the domain-code prefix before the first underscore in model_name (same
    grouping plot_results_domains.py used)."""
    csv_path = os.path.join(RESULTS_DIR, '{}_{}_{}.csv'.format(llm, dataset, direction))
    df = pd.read_csv(csv_path)
    df = df[~df['model_name'].apply(core_name).isin(exclude_files)]
    df = df[df[metric_column].notna()]

    pooled_values = df[metric_column].tolist()
    domain_group = df['model_name'].str.extract(r'^(.*?)_')[0]

    grouped = pd.DataFrame()
    for domain, sub in df.groupby(domain_group):
        grouped[domain] = pd.Series(sub[metric_column].reset_index(drop=True))

    return grouped, pooled_values


def plot_domain_boxplots(dataset=DATASET, direction=DIRECTION, metric_column=METRIC_COLUMN, metric_label=METRIC_LABEL):
    exclude_files = set(get_files_to_exclude(ITER_RESULTS_DIR, direction))
    llms = find_llms(RESULTS_DIR, dataset, direction)
    if not llms:
        print('No results csv found for dataset={} direction={} in {}'.format(dataset, direction, RESULTS_DIR))
        return

    fig, axes = plt.subplots(1, len(llms), figsize=(9 * len(llms), 7), sharey=True)
    if len(llms) == 1:
        axes = [axes]

    for ax, llm in zip(axes, llms):
        grouped, pooled_values = load_domain_scores(llm, dataset, direction, metric_column, exclude_files)
        mean_val = np.nanmean(pooled_values)

        palette = sns.blend_palette(COLOURS, n_colors=max(len(grouped.columns), 2))
        sns.boxplot(data=grouped, palette=palette, ax=ax)
        sns.swarmplot(data=grouped, color='black', size=4, ax=ax)
        ax.axhline(mean_val, color='#cb4335', linestyle='--', linewidth=1, label='Mean: {:.2f}'.format(mean_val))
        ax.legend()
        ax.set_ylim(0.4, 1)
        ax.set_xlabel('Domains', size=12)
        ax.tick_params(axis='x', labelsize=8, rotation=90)
        ax.set_ylabel('{} - {}'.format(metric_label, llm.upper()), size=12)

    plt.tight_layout()
    os.makedirs(PICS_DIR, exist_ok=True)
    base = os.path.join(PICS_DIR, '{} - {} - {}'.format(metric_label, dataset, direction))
    plt.savefig(base + '.png', dpi=300)
    plt.savefig(base + '.pdf')
    plt.savefig(base + '.svg')
    print('Saved {}.png and {}.pdf'.format(base, base))


if __name__ == "__main__":
    plot_domain_boxplots()
