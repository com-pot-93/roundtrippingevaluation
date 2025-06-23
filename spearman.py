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
        sqst = df['t2t_eval_2'].fillna(0).tolist()
        ebsm = df['m2m_eval_1'].fillna(0).tolist()
        fbsm= df['m2m_eval_2'].fillna(0).tolist()
        
        print("----- EBSM vs FBSM -----")
        corr, pval = spearmanr(fbsm, ebsm)
        print( corr, pval ,f)
        
        print("----- SMST vs SQST -----")
        corr, pval = spearmanr(smst, sqst)
        print( corr, pval ,f)
        
        print("----- FBSM vs SQST -----")
        corr, pval = spearmanr(fbsm, sqst)
        print( corr, pval ,f)
        
        print("----- FBSM vs SMST -----")
        corr, pval = spearmanr(fbsm, smst)
        print( corr, pval ,f)

        print("----- EBSM vs SQST -----")
        corr, pval = spearmanr(ebsm, sqst)
        print( corr, pval ,f)

        print("----- EBSM vs SMST -----")
        corr, pval = spearmanr(ebsm, smst)
        print( corr, pval ,f)










