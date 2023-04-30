import pandas as pd
import numpy as np


df = pd.read_pickle("normal.pkl")
df[['dist_to_mean','mean_opt']].to_csv('normal.csv')


def make_opt_plot(df):
	data = df['mean_opt']

	methods = [IRV,SRB,Borda, SRC, Plurality]
	scheme_str = [s.__name__ for s in schemes]
	num_meths = 5
	num_stats = 2

	fig, axs = plt.subplots(nrows=num_stats, ncols=num_meths+1, figsize=(num_meths*3,num_stats*3), width_ratios=[3]*num_meths+[.1])







df = pd.read_pickle("random.pkl")
df[['dist_to_mean','mean_opt']].to_csv('random.csv')