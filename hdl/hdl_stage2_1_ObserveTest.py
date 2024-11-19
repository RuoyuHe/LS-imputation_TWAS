# without corrected variance, train on imputed traits, test on observed
import os
import sys
print(sys.version)
import pdb
import pickle
import itertools
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as K
import statsmodels.api as sm
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from tensorflow.keras import Model
from tensorflow.keras.layers import Activation, Add, Input, Dense, ReLU, Reshape
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from wrappers import create_stage1, create_stage2, fit_stage2, tune_l2, stage2Tests, stage2Tests_C
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

def train_val_test_split(data, train_ratio, val_ratio, random_state):
	'''
		data: a pd.DataFrame of size n x p, first column is Y, the rest are X
	'''
	n = data.shape[0]
	train, val, test = np.split(data.sample(frac = 1, random_state = random_state), 
											[int(train_ratio * n), int((val_ratio + train_ratio) * n)])
	# if train.shape[0]<=150 or val.shape[0]<=150 or test.shape[0] <= 150:
	# 	return None
	y_train = train.iloc[:,0].to_numpy()
	y_val = val.iloc[:,0].to_numpy()
	y_test = test.iloc[:,0].to_numpy()
	x_train = train.iloc[:,1:].to_numpy()
	x_val = val.iloc[:,1:].to_numpy()
	x_test = test.iloc[:,1:].to_numpy()
	return {'y_train':y_train,
			'y_val':y_val,
			'y_test':y_test,
			'mean_x_train':x_train,
			'mean_x_val':x_val,
			'mean_x_test':x_test}

def TWAS(X1, X2, Y):
	design = sm.add_constant(X1)
	m = sm.OLS(Y, design)
	m_results = m.fit()
	p_g_X1 = m_results.pvalues[-1]
	#
	design = sm.add_constant(np.concatenate((X1.reshape(-1,1), X2.reshape(-1,1)), axis = 1))
	m = sm.OLS(Y, design)
	m_results = m.fit()
	p_g_X2 = m_results.f_pvalue
	p_nl_X2 = m_results.pvalues[-1]
	return p_g_X1, p_g_X2, p_nl_X2, m_results.params[1:]


# Defining the R script and loading the instance in Python
r = robjects.r
r['source']('/home/panwei/he000176/deepRIV/ImputedTraits/code/hdl_stage1_FUN_impute.R')

# Loading the function we have defined in R.
stage1_r = robjects.globalenv['stage1']

file_num = int(sys.argv[-1])
n1_samplesize = int(sys.argv[1])
runs = 70
start = (file_num-1)*runs + 1
end = min(file_num*runs + 1, 19697)
runs = end - start

# split_seeds = (ord('u'), ord('k'), ord('b'))
split_ratio = ({'train_ratio':0.5, 
				'val_ratio':0.1, 
				'random_state':ord('u')},
				{'train_ratio':0.4, 
				'val_ratio':0.1, 
				'random_state':ord('k')},
				{'train_ratio':0.3, 
				'val_ratio':0.1, 
				'random_state':ord('b')})
n2_split_ratio = ({'train_ratio':0.1, 
				'val_ratio':0.1, 
				'random_state':ord('u')},
				{'train_ratio':0.05, 
				'val_ratio':0.1, 
				'random_state':ord('k')},
				{'train_ratio':0.01, 
				'val_ratio':0.1, 
				'random_state':ord('b')})

training_params = ({'learning_rate':0.00001,
					'training_steps':300,
					'decay_steps':150,
					'decay_rate':1,
					'patience':15},
					{'learning_rate':0.00001,
					'training_steps':300,
					'decay_rate':1,
					'patience':15},
					{'learning_rate':0.00001,
					'training_steps':300,
					'batch_size':32,
					'decay_rate':1,
					'patience':15})

num_repeats = len(training_params)

data_path = '/home/panwei/shared/LSimputedHDLtrait/'
out_path = '/home/panwei/he000176/deepRIV/ImputedTraits/'
common_path = out_path + 'results/hdl/combine_p/TrainOn_n2_TestOn_n1/n1size_{}/common_{}.txt'.format(n1_samplesize,file_num)
IVa_path = out_path + 'results/hdl/combine_p/TrainOn_n2_TestOn_n1/n1size_{}/IVa_{}.txt'.format(n1_samplesize,file_num)
debug_path = out_path + 'results/debug/combine_p/{}.txt'.format(file_num)

hdl = pd.read_csv(data_path + 'original_hdl.csv', dtype = np.float32)
hdl.columns = ['id', 'hdl']
labels = hdl.hdl.values
labels_imputed = pd.read_csv(data_path + 'batchsize20000/yhat.txt', 
	dtype = np.float32, sep = ' ', header = None).values.squeeze()
# VarY_C = np.loadtxt('/home/panwei/he000176/deepRIV/ImputedTraits/data/boot_varEY/Var_Y.txt')

for gene_ind in range(start, end):
	gene_ind = int(gene_ind)
	print("gene_ind: {}".format(gene_ind))
	#converting it into r object for passing into r function
	with localconverter(robjects.default_converter + pandas2ri.converter):
		stage1_results = stage1_r(gene_ind)
		gene_name = robjects.conversion.rpy2py(stage1_results[0])
		#
		if gene_name[0] == "None":
			print(1, gene_ind)
			continue
		gene_names = gene_name[0]
		#
		ukb_snp = robjects.conversion.rpy2py(stage1_results[2])
		intercept = robjects.conversion.rpy2py(stage1_results[3])
		hatbetaX1 = robjects.conversion.rpy2py(stage1_results[4])
		intercept2 = robjects.conversion.rpy2py(stage1_results[5])
		hatbetaX2 = robjects.conversion.rpy2py(stage1_results[6])
		Y_ID = robjects.conversion.rpy2py(stage1_results[7])
		stage1_fstat = robjects.conversion.rpy2py(stage1_results[8])
		stage1_pval = robjects.conversion.rpy2py(stage1_results[9])
		stage1_X2_fstat = robjects.conversion.rpy2py(stage1_results[10])
		num_snps = robjects.conversion.rpy2py(stage1_results[11])
		stage1_sigma = robjects.conversion.rpy2py(stage1_results[12])
		Y = robjects.conversion.rpy2py(stage1_results[13])
		rs_id = robjects.conversion.rpy2py(stage1_results[14])
		num_individuals = Y_ID.shape[0]
		# if the test set is too small, skip
		if ukb_snp.shape[0] < 50000:
			print(2, gene_ind)
			continue

	# split n1 (complete) and n2 (genotype only) samples
	Y_ID = Y_ID.astype(np.int64)
	idx = np.array([np.where(hdl.id.values == Y_ID[i])[0][0] for i in range(Y_ID.shape[0])])
	Y = labels[idx]
	Y_imputed = labels_imputed[idx]
	n2_idx, n1_idx = train_test_split(np.arange(Y_ID.shape[0]), test_size=n1_samplesize, random_state=0)
	n1_ID = Y_ID[n1_idx]
	n2_ID = Y_ID[n2_idx]
	ukb_snp_n1 = ukb_snp[n1_idx,:]
	ukb_snp_n2 = ukb_snp[n2_idx,:]
	n1_labels = Y[n1_idx]
	n2_labels = Y_imputed[n2_idx]

	# impute missing genotype
	imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
	ukb_snp_n1 = imp_mean.fit_transform(ukb_snp_n1)
	ukb_snp_n2 = imp_mean.fit_transform(ukb_snp_n2)
	ukb_snp = imp_mean.fit_transform(ukb_snp)

	scaler_n1 = StandardScaler().fit(ukb_snp_n1)
	scaler_n2 = StandardScaler().fit(ukb_snp_n2)
	scaler_full = StandardScaler().fit(ukb_snp)
	# impute gene expression scaler_n1.transform()
	X_n1 = scaler_n1.transform(ukb_snp_n1) @ hatbetaX1 + intercept
	X_n2 = scaler_n2.transform(ukb_snp_n2) @ hatbetaX1 + intercept
	X2_n1 = scaler_n1.transform(ukb_snp_n1) @ hatbetaX2 + intercept2
	X2_n2 = scaler_n2.transform(ukb_snp_n2) @ hatbetaX2 + intercept2
	X = scaler_full.transform(ukb_snp) @ hatbetaX1 + intercept
	X2 = scaler_full.transform(ukb_snp) @ hatbetaX2 + intercept2
	# n1 complete TWAS-L, TWAS-LQ
	p_g_X1_n1, p_g_X2_n1, p_nl_X2_n1, _ = TWAS(X_n1, X2_n1, n1_labels)
	# now fit n1 and n2 separately, this is n2 only
	p_g_X1_n1n2, p_g_X2_n1n2, p_nl_X2_n1n2, _ = TWAS(X_n2, X2_n2, n2_labels)
	# n1+n2 complete TWAS-L, TWAS-LQ
	p_g_X1, p_g_X2, p_nl_X2, twaslq_coefs = TWAS(X, X2, Y)

	common_results = pd.DataFrame({"gene_names":gene_names,
						"pg_X1_n1":p_g_X1_n1,
						"pg_X2_n1":p_g_X2_n1,
						"pnl_X2_n1":p_nl_X2_n1,
						"pg_X1_n1n2":p_g_X1_n1n2,
						"pg_X2_n1n2":p_g_X2_n1n2,
						"pnl_X2_n1n2":p_nl_X2_n1n2,
						"pg_X1":p_g_X1,
						"pg_X2":p_g_X2,
						"pnl_X2":p_nl_X2,
						"stage1_fstat":stage1_fstat,
						"stage1_X2_fstat":stage1_X2_fstat,
						"num_snps":num_snps,
						"n1_size":n1_ID.shape[0],
						"n2_size":n2_ID.shape[0],
						"gene_ind":gene_ind
						}, index = [0])
	common_results.to_csv(common_path, mode = 'a', index = False, 
							sep = " ", header = not os.path.exists(common_path))
	with open(debug_path, "a") as f:
		f.write("{}\n".format(gene_ind))
		f.write("Common Done\n")

	IVa_pg_n1 = np.zeros(num_repeats)
	IVa_pnl_n1 = np.zeros(num_repeats)
	IVa_pg_n1n2 = np.zeros(num_repeats) # here n1n2 means train on n2 (imputed), test on n1
	IVa_pnl_n1n2 = np.zeros(num_repeats)
	IVa_pg_full = np.zeros(num_repeats)
	IVa_pnl_full = np.zeros(num_repeats)
	mu_max = np.zeros(num_repeats)
	mu_min = np.zeros(num_repeats)
	for j in range(num_repeats):
		# train test split
		tmp_data = pd.DataFrame(np.append((n1_labels - np.mean(n1_labels)).reshape(-1,1), X_n1.reshape(-1,1), axis = 1))
		data = train_val_test_split(tmp_data, **split_ratio[j])
		if data is None or np.unique(data['mean_x_test']).shape[0] < 2:
			IVa_pg_n1[j], IVa_pg_n1n2, IVa_pg_full = 100, 100, 100
			IVa_pnl_n1[j], IVa_pnl_n1n2, IVa_pnl_full = 100, 100, 100
			continue
		# n1
		l2 = 0
		rsp_n1 = create_stage2((1,),1, l2 = l2)
		_ = fit_stage2(rsp_n1, data['mean_x_train'], data['y_train'], data['mean_x_val'], data['y_val'],
						**training_params[j])
		# tests
		tmp = rsp_n1.predict(data['mean_x_test']).squeeze()
		IVa_pg_n1[j], _, IVa_pnl_n1[j], _, _, _, _, _ = stage2Tests(data['mean_x_test'], data['y_test'], tmp)
		# save model
		# if IVa_pnl_n1[j] <= 0.05/100:
		if False:
			model_path = out_path + 'models/hdl/combine_p/'
			rsp_n1.save(model_path + 'n1_{}_{}'.format(gene_ind, j))
		with open(debug_path, "a") as f:
			f.write("n1 {}: Done\n".format(j))

		# now fit n1 and n2 separately, this is n2 (imputed) only
		tmp_data = pd.DataFrame(np.append((n2_labels).reshape(-1,1), X_n2.reshape(-1,1), axis = 1))
		data = train_val_test_split(tmp_data, train_ratio=0.9, val_ratio=0.1, 
			random_state = int(np.random.sample(1)[0]*1000))
		# mu_max[j] = np.quantile(data['mean_x_test'], 0.975)
		# mu_min[j] = np.quantile(data['mean_x_test'], 0.025)
		# train
		l2 = 0
		rsp_n1n2 = create_stage2((1,),1, l2 = l2)
		_ = fit_stage2(rsp_n1n2, data['mean_x_train'], data['y_train'], data['mean_x_val'], data['y_val'],
						**training_params[j])
		# tests on n1
		tmp_data = pd.DataFrame(np.append((n1_labels - np.mean(n1_labels)).reshape(-1,1), X_n1.reshape(-1,1), axis = 1))
		data = train_val_test_split(tmp_data, train_ratio=0, val_ratio=0, 
			random_state = int(np.random.sample(1)[0]*1000))
		tmp = rsp_n1n2.predict(data['mean_x_test']).squeeze()
		IVa_pg_n1n2[j], _, IVa_pnl_n1n2[j], _, nonlinear_coef, nonlinear_se, linear_coef, linear_se = stage2Tests(data['mean_x_test'], data['y_test'], tmp)
		# save model
		# if IVa_pnl_n1n2[j] <= 0.05/100:
		if False:
			model_path = out_path + 'models/hdl/combine_p/'
			rsp_n1n2.save(model_path + 'n1n2_{}_{}'.format(gene_ind, j))
		with open(debug_path, "a") as f:
			f.write("n1n2 {}: Done\n".format(j))

		# full
		tmp_data = pd.DataFrame(np.append((n1_labels - np.mean(n1_labels)).reshape(-1,1), X_n1.reshape(-1,1), axis = 1))
		data = train_val_test_split(tmp_data, **n2_split_ratio[j])
		data['mean_x_train'] = np.concatenate((data['mean_x_train'], X[n2_idx].reshape(-1,1)), axis = 0)
		data['y_train'] = np.concatenate((data['y_train'], Y[n2_idx] - Y[n2_idx].mean(axis=0)), axis = 0)
		# train
		l2 = 0
		rsp_full = create_stage2((1,),1, l2 = l2)
		_ = fit_stage2(rsp_full, data['mean_x_train'], data['y_train'], data['mean_x_val'], data['y_val'],
						**training_params[j])
		# tests
		tmp = rsp_full.predict(data['mean_x_test']).squeeze()
		IVa_pg_full[j], _, IVa_pnl_full[j], _, _, _, _, _ = stage2Tests(data['mean_x_test'], data['y_test'], tmp)
		# save model
		# if IVa_pnl_full[j] <= 0.05/100:
		if False:
			model_path = out_path + 'models/hdl/combine_p/'
			rsp_full.save(model_path + 'full_{}_{}'.format(gene_ind, j))
		with open(debug_path, "a") as f:
			f.write("full {}: Done\n".format(j))

	gene_names = np.array(gene_names).reshape(-1,1)
	IVa_results = pd.DataFrame(gene_names)
	IVa_results = pd.concat([IVa_results, 
		pd.DataFrame(np.concatenate((IVa_pg_n1, IVa_pnl_n1,
									IVa_pg_n1n2, IVa_pnl_n1n2, 
									IVa_pg_full, IVa_pnl_full,
									[n1_ID.shape[0]], [n2_ID.shape[0]], [gene_ind])).reshape(1,-1))], 
		axis = 1)
	colnames = [['gene_names'], ['pg_n1']*num_repeats, ['pnl_n1']*num_repeats,
				['pg_n1n2']*num_repeats, ['pnl_n1n2']*num_repeats, 
				['pg_full']*num_repeats, ['pnl_full']*num_repeats,
				['n1_size'], ['n2_size'], ['gene_ind']]
	IVa_results.columns = list(itertools.chain(*colnames))
	IVa_results.to_csv(IVa_path, mode = 'a', index = False, 
							sep = " ", header = not os.path.exists(IVa_path))

