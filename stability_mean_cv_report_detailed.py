import os
import csv
from collections import defaultdict

GENERAL_SUMMARY_PATH = './stability/mean_general_summary.csv'
OUTPUT_PATH = './stability/mean_trustworthiness.csv'

# CV_pct consistency scale, in ascending order: (upper_bound, verdict, recommendation).
# CV < 2%: high consistency; 2% <= CV < 5%: moderate consistency; 5% <= CV < 10%: acceptable;
# CV >= 10%: substantial variability.
VERDICT_THRESHOLDS = [
    (2, "High consistency", "CV below 2%: outputs are highly consistent across trials."),
    (5, "Moderate consistency", "CV between 2% and 5%: outputs show moderate variability across trials."),
    (10, "Acceptable", "CV between 5% and 10%: outputs are usable, but note the higher variability."),
]
SUBSTANTIAL_VARIABILITY_RECOMMENDATION = "CV above 10%: outputs show substantial variability across trials."


def verdict_for(cv_pct):
    for threshold, verdict, recommendation in VERDICT_THRESHOLDS:
        if cv_pct < threshold:
            return verdict, recommendation
    return "Substantial variability", SUBSTANTIAL_VARIABILITY_RECOMMENDATION


def build_trustworthiness_report(general_summary_path=GENERAL_SUMMARY_PATH, output_path=OUTPUT_PATH):
    """Reads mean_general_summary.csv's median column and classifies each (llm, example, direction, data_metric) 
    combination's cv level from its CV_pct value, using the same thresholds values are reported in paper 
    (2,5,, and 10%)."""
    groups = defaultdict(dict)

    with open(general_summary_path, 'r', newline='') as csvfile:
        for row in csv.DictReader(csvfile):
            key = (row['llm'], row['example'], row['direction'], row['data_metric'])
            groups[key][row['stat']] = row['median']

    rows_out = []
    for (llm, example, direction, data_metric), stats in sorted(groups.items()):
        if 'CV_pct' not in stats:
            continue

        cv_pct = float(stats['CV_pct'])
        verdict, recommendation = verdict_for(cv_pct)

        rows_out.append({
            'llm': llm,
            'example': example,
            'direction': direction,
            'data_metric': data_metric,
            'CV_pct': cv_pct,
            'MedianCV_pct': stats.get('MedianCV_pct', ''),
            'CI_width': stats.get('CI_width', ''),
            'mean_median_diff_pct': stats.get('mean_median_diff_pct', ''),
            'verdict': verdict,
            'recommendation': recommendation,
        })

    fieldnames = [
        'llm', 'example', 'direction', 'data_metric', 'CV_pct',
        'MedianCV_pct', 'CI_width', 'mean_median_diff_pct',
        'verdict', 'recommendation',
    ]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    return output_path

if __name__ == "__main__":
    path = build_trustworthiness_report()
    print(f"Wrote {path}")
