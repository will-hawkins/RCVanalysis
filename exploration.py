import numpy as np
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import matplotlib.patches as mpatches
from matplotlib import colors
from matplotlib.colors import LinearSegmentedColormap
from elections import *
from voters import *

matplotlib.use('WebAgg')

"""
def sim(num_sim, N, A, d, method):
	dist_to_winner = np.zeros([num_sim,2])
	opt_winner = np.zeros(num_sim)
	np.random.seed(40)
	for i in range(num_sim):
		P = gen_voters_random(N,d)
		C = gen_voters_random(A,d)

		E = Election(P, C)
		E.preprocess()

		s, r, o = method(E).run()


		dist_to_winner[i] = E.dist_to_alt(o[0]).mean()/d**.5, np.median(E.dist_to_alt(o[0]))/d**.5

		opt_winner[i] = int(E.dist_matrix.sum(axis=0).argmin() == o[0])

	#breakpoint()
	return dist_to_winner, opt_winner"""


def sim(E, method):
	s, r, o = method(E).run()

	dist_to_mean = E.dist_to_alt(o[0]).mean()
	#dist_to_median= np.median(E.dist_to_alt(o[0]))/d**.5

	opt_mean = int(E.dist_matrix.sum(axis=0).argmin() == o[0])
	return dist_to_mean, opt_mean, o[0]



def sim_elections(num_sims, num_voters, num_alts, num_dims, schemes):

	scheme_str = [s.__name__ for s in schemes]
	index = pd.MultiIndex.from_product(
		[num_voters,num_alts,num_dims,scheme_str], names=['voters','alts','dims','scheme']	)

	df = pd.DataFrame(index=index,
					 columns=['dist_to_mean','mean_opt','winners'],
					 dtype=object)

	for i in num_voters:
		for j in num_alts:
			for k in num_dims:
				for l in scheme_str:
					for c in df.columns:
						df.loc[i,j,k,l][c] = np.empty(num_sims)


	for v in tqdm(num_voters):
		for a in tqdm(num_alts,leave=False):
			for d in num_dims:
				P = gen_voters_random(v,d)
				C = gen_voters_random(a,d)

				E = Election(P, C)
				E.preprocess()
				for s in range(num_sims):
					for l,scheme in enumerate(schemes):
						result = sim(E,scheme)
						row = df.loc[v,a,d,scheme_str[l]]
						row['dist_to_mean'][s] = result[0]/d
						row['mean_opt'][s] = result[1]
						row['winners'][s] = result[2]

	return df

if __name__ == '__main__':
	data = {}
	V = [10,100,500,1000,10000]
	DIMS = [1,2,3,5,10]
	C =[2,3,4,5,10]
	schemes = [IRV,SRB,Borda, SRC, Plurality]
	scheme_str = [s.__name__ for s in schemes]
	df = sim_elections(1000, V, C, DIMS, schemes)
	for i in V:
		for j in C:
			for k in DIMS:

				for l in scheme_str:
					df.loc[i,j,k,l]['dist_to_mean'] = df.loc[i,j,k,l]['dist_to_mean'].mean()
					df.loc[i,j,k,l]['mean_opt'] = df.loc[i,j,k,l]['mean_opt'].mean()

					agreement = np.array([len(schemes), len(schemes)])

	index = pd.MultiIndex.from_product(
		[V,C,DIMS], names=['voters','alts','dims']	)
	agreement = pd.DataFrame(index=index, dtype=object, columns=['M'])
	for i in V:
		for j in C:
			for k in DIMS:
				M = np.zeros([len(schemes),len(schemes)])
				for a, s1 in enumerate(schemes):
					for b, s2 in enumerate(schemes[:a]):
						M[a,b] = sum(df.loc[i,j,k,scheme_str[a]]['winners'] == df.loc[i,j,k,scheme_str[b]]['winners'])
				agreement.loc[i,j,k]['M'] = M
	df.to_pickle('uniform.pkl')
	agreement.to_pickle('uniform_agrmt.pkl')

def other():
	methods = [IRV,SRB,Borda, SRC, Plurality]
	meassures = ['mean', 'opt', 'median']
	num_meths = len(methods)
	num_stats = len(meassures)
	
	for ax,method in enumerate(methods):
		mean = np.zeros([len(DIMS),len(C)])
		opt = np.zeros([len(DIMS),len(C)])
		med = np.zeros([len(DIMS),len(C)])
		for i,d in enumerate(DIMS):
			for j,c in enumerate(C):
				D,O = sim(100, 1000, c, d, method)
				mean[i,j] = D[:,0].mean()
				med[i,j]  = D[:,1].mean()
				opt[i,j] = O.mean()

		print(opt)
		data[method.__name__] = {'mean': mean, "opt" : opt, "median": med}
		
	fig, axs = plt.subplots(nrows=num_stats, ncols=num_meths+1, figsize=(num_meths*3,num_stats*3), width_ratios=[3]*num_meths+[.1])

	
	combined = {key : np.empty([len(methods),len(DIMS),len(C)]) for key in meassures}

	for i,stat in enumerate(meassures):
		vmin = min([df[stat].min() for df in data.values()])
		vmax = max([df[stat].max() for df in data.values()])
		
		for j,method in enumerate(methods):

			dat = data[method.__name__][stat]

			dab = pd.DataFrame(dat, columns=C, index=DIMS)	
			sns.heatmap(dab, annot=False, yticklabels=True, vmin=vmin, vmax=vmax, linewidth=0.5, ax=axs[i][j], cbar=False)
			
			#axs[i,-1].set_title(method.__name__)

			combined[stat][j] = dat 

		fig.colorbar(axs[i][0].collections[0], cax=axs[i][-1])		
		axs[i,-1].yaxis.set_label_position("left")
		axs[i,-1].set_ylabel(stat)


	#breakpoint()
	



	### FROM: https://stackoverflow.com/questions/25812255/row-and-column-headers-in-matplotlibs-subplots ###
	cols = [method.__name__ for method in methods]
	rows = meassures


	pad = 5 # in points

	for ax, col in zip(axs[0,:-1], cols):
	    ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
	                xycoords='axes fraction', textcoords='offset points',
	                size='large', ha='center', va='baseline')

	for ax in axs[:,0]:
	    ax.set_ylabel('# of Dimensions')


	### END CITATION ###
	axs[-1,-1].yaxis.set_label_position("left")
	fig.tight_layout()
	fig.savefig(f"dims_alts.png")

	fig, axs = plt.subplots(1,3)
	myColors = ['k','b','y','g','r']
	cmap = LinearSegmentedColormap.from_list('Custom', myColors, len(myColors))

	for i, stat in enumerate(meassures):
		if i == 1:
			winner = pd.DataFrame((-combined[stat]).argmin(axis=0), columns=dab.columns, index=dab.index)
		else:
			winner = pd.DataFrame(combined[stat].argmin(axis=0), columns=dab.columns, index=dab.index)
		
		sns.heatmap(winner, annot=True,cmap=cmap, linewidths=.5,linecolor='lightgray', ax=axs[i])

		if i == 2:
			colorbar = axs[2].collections[0].colorbar
			colorbar.set_ticks(np.arange(.25,3.25,3/5))
			colorbar.set_ticklabels([m.__name__ for m in methods])


	fig.savefig("dims_alts_winners1.png")