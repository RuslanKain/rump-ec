import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set(rc={'figure.figsize':(11, 4)})
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
import time
from datetime import datetime
from tqdm import tqdm
from statsmodels.tsa.vector_ar.vecm import coint_johansen 
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tools.eval_measures import rmse
from matplotlib import pyplot
from pandas import DataFrame


class VARMA():

    def varma_model(self, train):
        model = VARMAX(endog=train)
        model_fit = model.fit(verbose=0)
        return model


    #Test relationship between different features
#tests the null hypothesis that the coefficients of past values in the regression equation is zero.

    def grangers_causation_matrix(self, data, variables, test='ssr_chi2test', verbose=False):    
        """Check Granger Causality of all possible combinations of the Time series.
        The rows are the response variable, columns are predictors. The values in the table 
        are the P-Values. P-Values lesser than the significance level (0.05), implies 
        the Null Hypothesis that the coefficients of the corresponding past values is 
        zero, that is, the X does not cause Y can be rejected.

        data      : pandas dataframe containing the time series variables
        variables : list containing names of the time series variables.
        """
        maxlag=12
        df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
        for c in df.columns:
            for r in df.index:
                test_result = grangercausalitytests(data[[r, c]], maxlag=maxlag, verbose=False)
                p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
                if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')
                min_p_value = np.min(p_values)
                df.loc[r, c] = min_p_value
        df.columns = [var + '_x' for var in variables]
        df.index = [var + '_y' for var in variables]
        return df

    # More attributes at: https://www.statsmodels.org/stable/generated/statsmodels.tsa.vector_ar.vecm.JohansenTestResult.html#rdb8d6a7c069c-1
    #from: https://www.machinelearningplus.com/time-series/vector-autoregression-examples-python/
    def cointegration_test(self, df, alpha=0.05): 
        """Perform Johanson's Cointegration Test and Report Summary"""
        out = coint_johansen(df,-1,5)
        d = {'0.90':0, '0.95':1, '0.99':2}
        traces = out.lr1
        cvts = out.cvt[:, d[str(1-alpha)]]
        def adjust(val, length= 6): return str(val).ljust(length)

        # Summary
        print('Name   ::  Test Stat > C(95%)    =>   Signif  \n', '--'*20)
        for col, trace, cvt in zip(df.columns, traces, cvts):
            print(adjust(col), ':: ', adjust(round(trace,2), 9), ">", adjust(cvt, 8), ' =>  ' , trace > cvt)