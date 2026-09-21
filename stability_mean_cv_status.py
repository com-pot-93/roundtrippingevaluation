import csv
from collections import defaultdict

TRUSTWORTHINESS_PATH = './stability/mean_trustworthiness.csv'
VERDICTS = ['High consistency', 'Moderate consistency', 'Acceptable', 'Substantial variability']


def load_verdict_counts(trustworthiness_path=TRUSTWORTHINESS_PATH):
    """{(llm, direction): {verdict: count}}, one count per (dataset, data_metric) row in
    trustworthiness.csv -- i.e. how many dataset+metric combinations landed in each verdict."""
    counts = defaultdict(lambda: defaultdict(int))

    with open(trustworthiness_path, 'r', newline='') as csvfile:
        for row in csv.DictReader(csvfile):
            key = (row['llm'], row['direction'])
            counts[key][row['verdict']] += 1

    return counts


def print_table(label, headers, rows):
    widths = [max(len(str(cell)) for cell in col) for col in zip(headers, *rows)]

    print(label)
    print(' '.join(h.ljust(w) for h, w in zip(headers, widths)))
    for row in rows:
        print(' '.join(str(cell).ljust(w) for cell, w in zip(row, widths)))
    print()


def print_verdict_count_table(trustworthiness_path=TRUSTWORTHINESS_PATH):
    """How many dataset+metric combinations fall into each consistency verdict, per pipeline
    (llm + direction) -- e.g. "anthropic's M2M pipeline has 7 High consistency, 2 Moderate
    consistency, and 1 Substantial variability combination". One table per direction, llm as
    columns and verdict as rows."""
    counts = load_verdict_counts(trustworthiness_path)
    if not counts:
        print('No data in {}.'.format(trustworthiness_path))
        return

    directions = sorted({direction for _, direction in counts})
    for direction in directions:
        llms = sorted(llm for llm, d in counts if d == direction)
        if not llms:
            continue

        rows = [
            [verdict] + [counts[(llm, direction)].get(verdict, 0) for llm in llms]
            for verdict in VERDICTS
        ]
        rows.append(['Total'] + [sum(counts[(llm, direction)].values()) for llm in llms])

        print_table(
            '{} pipeline: consistency verdict counts'.format(direction.upper()),
            [''] + llms,
            rows,
        )


if __name__ == "__main__":
    print_verdict_count_table()
