# roundtrippingevaluation

## Note

The implementation of the pipelines, along with the similarity computation, is available here: [Git](https://anonymous.4open.science/r/llm-round-trip-correctness-8ABD/README.md).

## Repository Overview

This repository provides the data and analysis scripts used during the evaluation of the LLMs.

Since executing the same prompts on the same data does not guarantee identical results, we provide all generated artifacts as produced during the evaluation. The evaluation consists of **10 independent iterations** for each pipeline, LLM, and dataset. This allows the results to be inspected, reproduced, and further analyzed without requiring the pipelines to be executed again.

The repository is organized into directories containing the generated data and intermediate results, as well as scripts used to calculate the statistics, tables, and figures reported in the evaluation.

## Directories

### `generated_artifacts`

Contains files generated during **10 independent runs** using the proposed pipelines.

Each file is a JSON object where the key is the filename and the value is an array containing the results from **iterations 0–9**. Each iteration contains both the generated model output and the corresponding generated text.

The generated artifacts serve as the basis for the iteration-level similarity evaluation in `iter_results`.

### `iter_results`

Contains similarity measurements calculated independently for **each of the 10 iterations**.

These results preserve the individual measurements for all 10 runs and are used as input for the stability analysis and the calculation of average similarity values.

### `results`

Contains the average similarity values calculated across **all 10 iterations**.

These results are used for the performance analysis and other analyses that operate on the aggregated similarity values.

### `stability`

Contains the results of the stability evaluation across the **10 iterations**, including various statistical measures used to assess the consistency of the results across repeated runs.

### `check`

Contains values calculated during the comparison of texts and models across all **5 datasets**. These metrics are used to evaluate the consistency of the proposed pipelines across datasets.

## Tables

### Completion Rate (Section 4.2)

`completion_rate.py`

Calculates the completion rate used to produce Table 3.

### Stability Across Runs (Section 4.3)

`stability_calculate_all_values.py`

Requires `iter_results/` and creates files in `stability/`.

Calculates statistics for each description across the **10 iterations**. Files with fewer than two iterations are excluded from the analysis.

`stability_median_general.py`

Requires `stability/` and creates `stability/general_summary.csv`.

Calculates the median across descriptions for each statistic, separately for each LLM, dataset, pipeline direction, and similarity metric.

`stability_median_cv_report_detailed.py`

Requires `general_summary.csv` and creates `trustworthiness.csv`.

Takes the median coefficient of variation (CV) values and assigns a corresponding consistency status:

- `< 2%`: High consistency
- `2–5%`: Moderate consistency
- `5–10%`: Acceptable consistency
- `>= 10%`: Substantial variability

`stability_median_cv_status.py`

Requires `trustworthiness.csv`.

Reports, for each pipeline direction, the number of dataset/metric combinations that fall into each consistency category.

`stability_median_cv_summary.py`

Requires `stability/`.

Prints the median CV values as tables, with one table for each pipeline direction and LLM. These results correspond to Tables 4 and 5 and are calculated based on the **10 evaluation iterations**.

#### Mean-Based Comparison

The corresponding scripts for the mean-based analysis have the same functionality as their median-based counterparts, but calculate and report mean values instead:

- `stability_mean_general.py`
- `stability_mean_cv_report_detailed.py`
- `stability_mean_cv_status.py`
- `stability_mean_cv_summary.py`

All mean-based stability statistics are calculated from the results of the **10 iterations**.

### Performance Overview (Section 4.4)

`performance_absolute.py`

Generates the absolute performance results presented in Tables 6 and 7. The underlying similarity results are based on the **10 evaluation iterations**.

`performance_friedman_detailed.py`

Requires `results/` and `iter_results/`.

Performs a detailed Friedman-test analysis for each pipeline direction, dataset, and similarity metric. It reports the number of files, the LLMs being compared, the Friedman statistic (`χ²`), the p-value, and the mean rank of each LLM.

The analysis uses the similarity results obtained from the **10 iterations**.

`performance_friedman.py`

Requires `results/` and `iter_results/`.

Performs the Friedman test and produces a summary containing the Friedman statistic (`χ²`), p-value, winning LLM, and significance (`p < 0.05`) for each pipeline direction, dataset, and similarity metric. These results are presented in Table 8.

The statistical comparison is based on the evaluation results from **10 iterations**.

`performance_nemenyi_detailed.py`

Requires `results/` and `iter_results/`.

Performs the Nemenyi post-hoc analysis and reports the higher- and lower-ranked LLMs, their mean scores, the raw score difference, the Nemenyi p-value, effect magnitude, significance, and the corresponding Friedman p-value.

The analysis is based on the results obtained across the **10 evaluation iterations**.

`performance_nemenyi_summary.py`

Requires `results/` and `iter_results/`.

For cases where the Friedman test is significant, summarizes the Nemenyi pairwise comparisons. It reports the pairwise p-value matrix, the mean rank of each LLM together with the critical difference, and the pairwise rank differences.

It also reports the number of significant pairwise comparisons overall and for each LLM pair. The comparisons are based on the **10 evaluation iterations**.

`performance_kendall_summary.py`

Requires `results/` and `iter_results/`.

Calculates Kendall's W from the Friedman statistic to quantify the degree of agreement among the LLM rankings across files, based on the results from the **10 iterations**.

### RTC (Section 4.5)

`rtc_correlation.py`

Calculates the RTC correlations presented in Tables 9, 10, and 11 using the evaluation results obtained from the **10 iterations**.

## Figures

### Model-to-Text Overlap (Section 4.1)

`plot_text_model_bars.py`

Visualizes the average model-to-text evaluation using precision and recall across all datasets. The averages are calculated from the results of the **10 evaluation iterations**.

### Performance Overview (Section 4.4)

`plot_similarity_trends.py`

Visualizes the average similarity scores across datasets, pipelines, and directions.

The resulting plots show the mean similarity trends for the M2M and T2T pipelines and correspond to Figures 7 and 8. The plotted averages are calculated across the **10 evaluation iterations**.

### RTC Spread Across Domains (Section 4.6)

The following scripts generate box plots showing the distribution of similarity measurements across domains and business areas. These visualizations correspond to Figures 9–12 and are based on the evaluation results from the **10 iterations**.

`plot_domain_boxplots_all_llms.py`

Generates domain-level box plots across all LLMs using the results from the **10 evaluation iterations**.

`plot_mad_boxplots_all_llms.py`

Generates box plots based on the mean absolute deviation (MAD) across all LLMs using the results from the **10 evaluation iterations**.


