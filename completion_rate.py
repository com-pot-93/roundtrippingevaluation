import os
import pandas as pd
import json
import numpy as np

dir_path = "./iter_results"
results = []
for filename in os.listdir(dir_path):
    if not filename.endswith(".json"):
        continue
    file_path = os.path.join(dir_path, filename)
    with open(file_path, "r") as infile:
        data = json.load(infile)
    number_of_files = len(data)
    count_10_iterations = sum(len(iterations) == 10 for iterations in data.values())
    count_0_iterations = sum(len(iterations) == 0 for iterations in data.values())
    count_1_iterations = sum(len(iterations) == 1 for iterations in data.values())
    iterations_per_file = [[file_name, len(iterations)] for file_name, iterations in data.items() ]
    results.append([ filename, number_of_files, count_10_iterations, count_0_iterations, count_1_iterations ])

for entry in results:
    print(entry)


