import pandas as pd
import numpy as np
import os
import json

dir_path = './iter_results'
files_to_iterate = os.listdir(dir_path)
for f in files_to_iterate:
    #print(f)
    #if "t2t" in f:
    #    if "gemini" in f:
    print(f)
    with open(os.path.join(dir_path, f), "r") as infile:
        data = json.load(infile)
        all_means = []
            # Loop over keys (0, 1, 2) and collect the data for each key in all files
        for key in range(3):
            # Collect the data for this key across all files
            combined_data = []
            for file_name, values in data.items():
                try:
                    combined_data.append([file_name, key] + values[str(key)])
                except:
                    combined_data.append([file_name, key] + [0,0,0,0])
            #    pass
            # Create a DataFrame for the collected data
            df = pd.DataFrame(combined_data, columns=['File Name', 'Key', 'Value 1', 'Value 2', 'Value 3', 'Value 4'])
            means = df[['Value 1', 'Value 2', 'Value 3', 'Value 4']].mean()
            means = means.tolist()
            all_means.append(means)

        new_df = pd.DataFrame(all_means)

        test_res = []
        for column in new_df:
            temp = new_df[column].tolist()
            range_value = round(np.ptp(temp),2)
            variance = np.var(temp,ddof=0)
            std_deviation = np.std(temp,ddof=0)
            test_res.append([range_value,variance,std_deviation])

        res_df = pd.DataFrame(test_res).transpose()
        res_df.index = ['range','variance','std. deviation']

        combined_df = pd.concat([new_df, res_df])
        print(combined_df)

                    # Write the DataFrame to the corresponding sheet
#        df.to_excel(#writer, sheet_name=f'Key {key}', index=False)

#    print("Excel file saved as 'output_separate_sheets.xlsx'")
