import os
import csv
import json
import logging

import numpy as np
from scipy.stats import t as t_dist

VALID_METRICS = {"text_eval_1", "text_eval_2", "model_eval_1", "model_eval_2"}


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


def get_similarity_series(report_path, metric):
    """Converts an iter_results report json into {filename: [sim_iter0, sim_iter1, ...]},
    picking one of text_eval_1, text_eval_2, model_eval_1, model_eval_2 per iteration."""
    if metric not in VALID_METRICS:
        raise ValueError("metric must be one of {}, got {!r}".format(sorted(VALID_METRICS), metric))

    with open(report_path, "r") as infile:
        data = json.load(infile)

    series = {}
    for file_name, iterations in data.items():
        ordered_iterations = sorted(iterations.items(), key=lambda item: int(item[0]))
        series[file_name] = [scores[metric] for _, scores in ordered_iterations]

    return series


def compute_description_stats(scores):
    """Per-description stability statistics (mean/median/SD/CV/CI) over one description's
    iteration scores. Returns None if fewer than 2 numeric scores are available."""
    numeric_scores = [s for s in scores if isinstance(s, (int, float))]
    n = len(numeric_scores)
    if n < 2:
        return None

    scores_arr = np.array(numeric_scores, dtype=float)
    mean_val = np.mean(scores_arr)
    median_val = np.median(scores_arr)
    std_val = np.std(scores_arr, ddof=1)  # sample SD (N-1)
    min_val = np.min(scores_arr)
    max_val = np.max(scores_arr)
    range_val = max_val - min_val

    cv_pct = (std_val / mean_val) * 100 if mean_val != 0 else np.nan

    abs_devs_from_median = np.abs(scores_arr - median_val)
    mad_val = np.median(abs_devs_from_median)
    median_cv_pct = (mad_val / median_val) * 100 if median_val != 0 else np.nan

    sem_val = std_val / np.sqrt(n)
    t_crit = t_dist.ppf(0.975, n - 1)
    ci_margin = t_crit * sem_val
    ci_lower = mean_val - ci_margin
    ci_upper = mean_val + ci_margin
    ci_width = ci_upper - ci_lower

    mean_median_diff = np.abs(mean_val - median_val)
    mean_median_pct = (mean_median_diff / median_val) * 100 if median_val != 0 else np.nan

    return {
        'n': n,
        'range': range_val,
        'mean': mean_val,
        'median': median_val,
        'SD': std_val,
        'CV_pct': cv_pct,
        'MedianCV_pct': median_cv_pct,
        'CI_95_lower': ci_lower,
        'CI_95_upper': ci_upper,
        'CI_width': ci_width,
        'mean_median_diff_pct': mean_median_pct,
    }


def compute_report_stability(report_path, exclude_files, logger):
    """Per-description stats for every metric in a single iter_results report."""
    rows = []
    for metric in sorted(VALID_METRICS):
        series = get_similarity_series(report_path, metric)
        for description, scores in series.items():
            if core_name(description) in exclude_files:
                continue

            stats = compute_description_stats(scores)
            if stats is None:
                logger.warning(f"Skipping '{description}' ({metric}) in {report_path}: fewer than 2 numeric scores")
                continue

            rows.append({'description': description, 'metric': metric, **stats})

    return rows


def build_stability_reports(iter_results_dir='../iter_results', stability_dir='./stability'):
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("StabilityLogger")
    logger.setLevel(logging.INFO)

    os.makedirs(stability_dir, exist_ok=True)

    # Files to exclude only depend on direction, not on which report they end up scoring, so
    # this scans iter_results once per direction instead of once per report file.
    exclude_by_direction = {
        direction: set(get_files_to_exclude(iter_results_dir, direction))
        for direction in ('m2m', 't2t')
    }

    fieldnames = [
        'description', 'metric', 'n', 'range', 'mean', 'median', 'SD',
        'CV_pct', 'MedianCV_pct', 'CI_95_lower', 'CI_95_upper', 'CI_width', 'mean_median_diff_pct',
    ]

    for filename in os.listdir(iter_results_dir):
        if not filename.endswith('.json'):
            continue

        if filename.endswith('_m2m.json'):
            direction = 'm2m'
        elif filename.endswith('_t2t.json'):
            direction = 't2t'
        else:
            logger.warning(f"Skipping {filename}: direction not recognized (expected _m2m.json or _t2t.json)")
            continue

        report_path = os.path.join(iter_results_dir, filename)
        rows = compute_report_stability(report_path, exclude_by_direction[direction], logger)

        csv_path = os.path.join(stability_dir, filename.replace('.json', '.csv'))
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        logger.info(f"Wrote {csv_path} ({len(rows)} rows)")


if __name__ == "__main__":
    build_stability_reports()
