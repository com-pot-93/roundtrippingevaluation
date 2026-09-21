import os
import csv
from collections import defaultdict

import numpy as np

STABILITY_DIR = './stability'
OUTPUT_FILENAME = 'mean_general_summary.csv'

STAT_COLUMNS = ['n', 'range', 'mean', 'median', 'SD', 'CV_pct', 'MedianCV_pct', 'CI_width', 'mean_median_diff_pct']


def parse_report_stem(stem):
    """"report_anthropic_domain_m2m" -> ("anthropic", "domain", "m2m")."""
    parts = stem.split('_')
    if parts and parts[0] == 'report':
        parts = parts[1:]
    if len(parts) < 3:
        return None, None, None
    return parts[0], '_'.join(parts[1:-1]), parts[-1]


def load_stability_csv(csv_path):
    """Groups a per-description stability csv's rows by the text/model metric column."""
    grouped = defaultdict(list)
    with open(csv_path, 'r', newline='') as csvfile:
        for row in csv.DictReader(csvfile):
            grouped[row['metric']].append(row)
    return grouped


def summarize(rows):
    """Median/min/max across descriptions for each stat column, given rows from one report+metric."""
    summary = {}
    for stat in STAT_COLUMNS:
        values = np.array([float(row[stat]) for row in rows if row[stat] not in ('', 'nan')], dtype=float)
        values = values[~np.isnan(values)]
        summary[stat] = (np.mean(values), np.min(values), np.max(values)) if len(values) else (np.nan, np.nan, np.nan)
    return summary


def build_general_summary(stability_dir=STABILITY_DIR, output_filename=OUTPUT_FILENAME):
    output_path = os.path.join(stability_dir, output_filename)

    rows_out = []
    for filename in sorted(os.listdir(stability_dir)):
        # Skip our own output so a re-run doesn't try to summarize the general summary itself.
        if not filename.endswith('.csv') or not filename.startswith('report_'):
            continue

        stem = filename[:-len('.csv')]
        llm, example, direction = parse_report_stem(stem)

        grouped = load_stability_csv(os.path.join(stability_dir, filename))
        for data_metric, rows in grouped.items():
            for stat, (median_val, min_val, max_val) in summarize(rows).items():
                rows_out.append({
                    'report': stem,
                    'llm': llm,
                    'example': example,
                    'direction': direction,
                    'data_metric': data_metric,
                    'stat': stat,
                    'median': median_val,
                    'min': min_val,
                    'max': max_val,
                })

    fieldnames = ['report', 'llm', 'example', 'direction', 'data_metric', 'stat', 'median', 'min', 'max']
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    return output_path


if __name__ == "__main__":
    path = build_general_summary()
    print(f"Wrote {path}")
