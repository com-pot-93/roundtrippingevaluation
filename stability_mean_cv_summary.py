import csv
from collections import defaultdict

GENERAL_SUMMARY_PATH = './stability/mean_general_summary.csv'

# SMST=text_eval_1 (sts_bert), SQST=text_eval_2 (text_similarity_alternative),
# EBSM=model_eval_1 (calculate_similarity union-weighted), FBSM=model_eval_2 (binarized-weight)
METRIC_LABELS = {
    'text_eval_1': 'SMST',
    'text_eval_2': 'SQST',
    'model_eval_1': 'EBSM',
    'model_eval_2': 'FBSM',
}
METRIC_ORDER = ['text_eval_1', 'text_eval_2', 'model_eval_1', 'model_eval_2']

# median CV_pct thresholds, in ascending order: (upper_bound, verdict, recommendation)
VERDICT_THRESHOLDS = [
    (5, "EXCELLENT", "Means are highly reliable. Proceed with Friedman test."),
    (10, "GOOD", "Means are reliable. Proceed with Friedman test."),
    (15, "ACCEPTABLE", "Means are usable, but note the higher variability in your paper."),
]
POOR_RECOMMENDATION = "High variability. Consider more runs, or use medians/nonparametric tests instead of means."


def verdict_for(median_cv):
    for threshold, verdict, recommendation in VERDICT_THRESHOLDS:
        if median_cv < threshold:
            return verdict, recommendation
    return "POOR", POOR_RECOMMENDATION


def load_median_cv_by_group(general_summary_path=GENERAL_SUMMARY_PATH):
    """{(llm, direction): {example: {data_metric: median_CV_pct}}}, read from general_summary.csv."""
    tables = defaultdict(lambda: defaultdict(dict))

    with open(general_summary_path, 'r', newline='') as csvfile:
        for row in csv.DictReader(csvfile):
            if row['stat'] != 'CV_pct':
                continue
            key = (row['llm'], row['direction'])
            tables[key][row['example']][row['data_metric']] = float(row['median'])

    return tables


def print_table(label, table):
    columns = [m for m in METRIC_ORDER if any(m in row for row in table.values())]
    headers = ['Dataset'] + [METRIC_LABELS[m] for m in columns]
    rows = [[example] + ['{:.2f}'.format(table[example][m]) if m in table[example] else ''
                          for m in columns]
            for example in sorted(table)]

    widths = [max(len(str(cell)) for cell in col) for col in zip(headers, *rows)]

    print(label)
    print(' '.join(h.ljust(w) for h, w in zip(headers, widths)))
    for row in rows:
        print(' '.join(str(cell).ljust(w) for cell, w in zip(row, widths)))
    print()


def print_trustworthiness_tables(general_summary_path=GENERAL_SUMMARY_PATH):
    """One CV_pct table per (llm, direction) found in general_summary.csv: rows = datasets,
    columns = SMST/SQST/EBSM/FBSM, matching the "Table 3: Evaluation of M2M pipeline" layout."""
    tables = load_median_cv_by_group(general_summary_path)

    for (llm, direction), table in sorted(tables.items()):
        print_table('Table: {} pipeline, {}'.format(direction.upper(), llm), table)


if __name__ == "__main__":
    print_trustworthiness_tables()
