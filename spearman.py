from scipy.stats import spearmanr
import pandas as pd
import os
# assign directory
directory = './results'

for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    if os.path.isfile(f):
        #print(f)
        df = pd.read_csv(f)
        smst = df['t2t_eval_1'].fillna(0).tolist()
        #sqst = df['t2t_eval_2'].fillna(0).tolist()
        #ebsm = df['m2m_eval_1'].fillna(0).tolist()
        fbsm= df['m2m_eval_2'].fillna(0).tolist()
        corr, pval = spearmanr(z, y)
        print( corr, pval ,f)









