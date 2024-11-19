import os
import sys
print(sys.version)
import pdb
import pickle
import shutil
import atexit
import logging
import traceback
import itertools
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as K
import statsmodels.api as sm
import statsmodels.formula.api as smf
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from tensorflow.keras import Model
from tensorflow.keras.layers import Activation, Add, Input, Dense, ReLU, Reshape
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
# from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from wrappers import create_stage1, create_stage2, fit_stage2, tune_l2, stage2Tests, stage2Tests_C
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import utils
from utils import SimpleImputer
# from importlib import reload


def TWAS(X1, Y, X2=None, cov=None):
    if cov is not None:
        tmp_data = cov.copy()
        tmp_data['y'] = Y.copy()
        tmp_data['sex'] = pd.Categorical(tmp_data['sex'].astype(int))
        model = smf.ols('y ~ pc1 + pc2 + pc3 + pc4 + pc5 \
            + pc6 + pc7 + pc8 + pc9 + pc10 + pc11 + pc12 + pc13 \
            + pc14 + pc15 + pc16 + pc17 + pc18 + pc19 + pc20 + sex + age + age2 + sex:age + sex:age2', 
        data=tmp_data).fit()
        Y = model.resid.copy()
    #
    design = sm.add_constant(X1)
    m = sm.OLS(Y, design)
    m_results = m.fit()
    p_g_X1 = m_results.pvalues[-1]
    #
    if X2 is not None:
        design = sm.add_constant(np.concatenate((X1.reshape(-1,1), X2.reshape(-1,1)), axis = 1))
        m = sm.OLS(Y, design)
        m_results = m.fit()
        p_g_X2 = m_results.f_pvalue
        p_nl_X2 = m_results.pvalues[-1]
    else:
        p_g_X2 = 100
        p_nl_X2 = 100
    return p_g_X1, p_g_X2, p_nl_X2, m_results.params[1:]


file_num = int(sys.argv[-1])
true_model = sys.argv[1] 

out_path = '/home/panwei/he000176/deepRIV/ImputedTraits/simulations/'
IVa_path = out_path + 'results/for_revision/delivr_{}_{}.txt'.format(true_model,file_num)
# model_path = out_path + 'models/realdata_sim/'
debug_path = out_path + 'results/debug/{}_{}.txt'.format(true_model,file_num)

logging.basicConfig(
    filename = debug_path + '.log',  # File to write logs to
    level = logging.DEBUG,  # Level of logging
    format = '%(asctime)s - %(levelname)s - %(message)s',  # Format of logs
    datefmt = '%Y-%m-%d %H:%M:%S'  # Date format in logs
)


data_path = '/home/panwei/he000176/deepRIV/ImputedTraits/data/'
gene_inds = np.loadtxt(data_path + 'gene_ind.txt', dtype = np.int32)


split_ratio = ({'test_ratio':0, 
                'val_ratio':0.1, 
                'random_state':ord('u')},
                {'test_ratio':0, 
                'val_ratio':0.1, 
                'random_state':ord('k')},
                {'test_ratio':0, 
                'val_ratio':0.1, 
                'random_state':ord('b')})
adsp_split_ratio = ({'test_ratio':0.4, 
                'val_ratio':0.1, 
                'random_state':ord('u')},
                {'test_ratio':0.5, 
                'val_ratio':0.1, 
                'random_state':ord('k')},
                {'test_ratio':0.55, 
                'val_ratio':0.1, 
                'random_state':ord('b')})

training_params = ({'learning_rate':0.00033,
                    'training_steps':999,
                    'decay_steps':150,
                    'batch_size':2048,
                    'decay_rate':1,
                    'patience':15},
                    {'learning_rate':0.00033,
                    'training_steps':999,
                    'batch_size':1024,
                    'decay_rate':1,
                    'patience':15},
                    {'learning_rate':0.00033,
                    'training_steps':999,
                    'batch_size':512,
                    'decay_rate':1,
                    'patience':15})

adsp_training_params = ({'learning_rate':0.00033,
                    'training_steps':999,
                    'decay_steps':150,
                    'decay_rate':1,
                    'patience':15},
                    {'learning_rate':0.000033,
                    'training_steps':999,
                    'decay_rate':.95,
                    'patience':15},
                    {'learning_rate':0.00033,
                    'training_steps':999,
                    'batch_size':32,
                    'decay_rate':.67,
                    'patience':15})



imp_mean = SimpleImputer(strategy='most_frequent')

ID_path = data_path + '/for_revision/effect_snps_ID.txt'
ref_ID = np.loadtxt(ID_path, dtype = np.int32)

for i in range(gene_inds.shape[0]):
    g = gene_inds[i]
    print(g)
    logging.info(f"Processing gene index: {g}")
    snp_path = data_path + f'{g}_bed.txt'
    ID_path = data_path + f'{g}_ID.txt'

    if not(os.path.exists(snp_path) and os.path.exists(ID_path)):
        continue
        
    ukb_snp = pd.read_csv(snp_path).values
    ukb_snp = imp_mean.fit_transform(ukb_snp)
    ID = np.loadtxt(ID_path, dtype = np.int32)
    idx = np.array([np.where(ID == ref_ID[i])[0][0] for i in range(ref_ID.shape[0])])
    ukb_snp = ukb_snp[idx]
    
    X = np.loadtxt(data_path + f'/for_revision/rep_{file_num}/{g}_simX.txt')
    if true_model == 'typeI':
        Y = np.loadtxt(data_path + f'/for_revision/rep_{file_num}/typeI_Y.txt')
        Y_imputed = np.loadtxt(data_path + f'/for_revision/rep_{file_num}/imputed_typeIY.txt')
    else:
        Y = np.loadtxt(data_path + f'/for_revision/rep_{file_num}/Y.txt')
        Y_imputed = np.loadtxt(data_path + f'/for_revision/rep_{file_num}/imputed_Y.txt')

    sim_idx = np.loadtxt(data_path + f'/for_revision/rep_{file_num}/simulation_idx.txt', dtype = np.int32)
    ukb_snp = ukb_snp[sim_idx,:]
    X = X[sim_idx]
    Y = Y[sim_idx]
    
    stage1_idx = np.arange(1000)
    stage2_idx = np.arange(1000,Y.shape[0])

    stage1_X = X[stage1_idx].copy()
    stage1_Z = ukb_snp[stage1_idx,:].copy()
    stage1_Z = imp_mean.fit_transform(stage1_Z)
    stage1_Z = StandardScaler().fit_transform(stage1_Z)
    stage1_X = StandardScaler().fit_transform(stage1_X.reshape(-1,1))
    design = sm.add_constant(stage1_Z)
    stage1 = sm.OLS(stage1_X, design)
    stage1 = stage1.fit()

    ukb_snp = ukb_snp[stage2_idx,:]
    X = X[stage2_idx]
    Y = Y[stage2_idx]
    Y_imputed = Y_imputed[stage2_idx]

    observed_idx, imputed_idx = train_test_split(np.arange(ukb_snp.shape[0]), test_size=39000, random_state=0)
    observed_idx.sort()
    imputed_idx.sort()
    
    obs_snp = ukb_snp[observed_idx,]
    obs_X = StandardScaler().fit_transform(imp_mean.fit_transform(obs_snp)) @ stage1.params[1:]
    obs_Y = Y[observed_idx]
    
    logging.debug(f"TWAS-L training started")
    twas_X1_obs, _, _, _ = TWAS(obs_X, StandardScaler().fit_transform(obs_Y.reshape(-1,1)).squeeze())
    
    imp_snp = ukb_snp[imputed_idx,]
    # imp_Y = Y[imputed_idx]
    imp_Y_imputed = Y_imputed[imputed_idx]
    
    j=0
    
    ########## fit and test on obs data
    data = utils.split_scale([obs_snp], [obs_Y],stage1.params[1:],0,['obs'], **adsp_split_ratio[j])
    l2 = 0
    rsp_n1 = create_stage2((1,),1, l2 = l2)

    logging.debug(f"DeLIVR (Obs) training started")

    _ = fit_stage2(rsp_n1, data['obs']['train']['x'], data['obs']['train']['y'], 
        data['obs']['val']['x'], data['obs']['val']['y'], **adsp_training_params[j])

    logging.debug(f"DeLIVR (Obs) training finished. Testing started")

    # tests
    tmp = rsp_n1.predict(data['obs']['test']['x']).squeeze()
    IVa_pg_obs, _, IVa_pnl_obs, _, _, _, _, _ = stage2Tests(data['obs']['test']['x'], 
        data['obs']['test']['y'], tmp)
    logging.debug(f"DeLIVR (Obs) training finished. Testing finished")
    
    ########## fit on imputed and test on obs data
    data = utils.split_scale([imp_snp], [imp_Y_imputed],stage1.params[1:],0,['imp'],
                             **split_ratio[j])
    l2 = 0
    rsp_n1n2 = create_stage2((1,),1, l2 = l2)

    logging.debug(f"DeLIVR (imp) training started")

    _ = fit_stage2(rsp_n1n2, data['imp']['train']['x'], data['imp']['train']['y'], 
        data['imp']['val']['x'], data['imp']['val']['y'], **training_params[j])

    logging.debug(f"DeLIVR (imp) training finished. Testing started")

    # tests
    tmp = rsp_n1n2.predict(obs_X).squeeze()
    IVa_pg_imp, _, IVa_pnl_imp, _, _, _, _, _ = stage2Tests(StandardScaler().fit_transform(obs_X.reshape(-1,1)), 
                                                            StandardScaler().fit_transform(obs_Y.reshape(-1,1)).squeeze(), 
                                                            tmp)
    
    ########## fit  and test on complete data
    data = utils.split_scale([ukb_snp], [Y],stage1.params[1:],0,['comp'],
                             **adsp_split_ratio[j])
    l2 = 0
    rsp_all = create_stage2((1,),1, l2 = l2)

    logging.debug(f"DeLIVR (comp) training started")

    _ = fit_stage2(rsp_all, data['comp']['train']['x'], data['comp']['train']['y'], 
        data['comp']['val']['x'], data['comp']['val']['y'], **training_params[j])

    logging.debug(f"DeLIVR (comp) training finished. Testing started")

    # tests
    tmp = rsp_all.predict(data['comp']['test']['x']).squeeze()
    IVa_pg_comp, _, IVa_pnl_comp, _, _, _, _, _ = stage2Tests(data['comp']['test']['x'], 
                                                            data['comp']['test']['y'], 
                                                            tmp)
    ########### save results
    IVa_results = pd.DataFrame(np.array(g).reshape(-1,1))
    IVa_results = pd.concat([IVa_results, 
        pd.DataFrame(np.concatenate(([twas_X1_obs], [IVa_pg_obs], [IVa_pnl_obs],
                                     [IVa_pg_imp], [IVa_pnl_imp], 
                                     [IVa_pg_comp], [IVa_pnl_comp],
                                    [obs_Y.shape[0]], [imp_Y_imputed.shape[0]], [Y.shape[0]])).reshape(1,-1))], 
        axis = 1)
    colnames = ['gene_ind','twasl_obs','pg_obs', 'pnl_obs',
                'pg_imp', 'pnl_imp',
                'pg_comp', 'pnl_comp',
                'obs_size', 'imp_size', 'comp_size']
    IVa_results.columns = colnames
    IVa_results.to_csv(IVa_path, mode = 'a', index = False, 
                            sep = " ", header = not os.path.exists(IVa_path))
    