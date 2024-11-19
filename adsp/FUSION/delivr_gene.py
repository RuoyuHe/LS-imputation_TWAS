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


def TWAS(X1, X2, Y, cov=None):
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
    design = sm.add_constant(np.concatenate((X1.reshape(-1,1), X2.reshape(-1,1)), axis = 1))
    m = sm.OLS(Y, design)
    m_results = m.fit()
    p_g_X2 = m_results.f_pvalue
    p_nl_X2 = m_results.pvalues[-1]
    return p_g_X1, p_g_X2, p_nl_X2, m_results.params[1:]


# Defining the R script and loading the instance in Python
r = robjects.r
r['source']('/home/panwei/he000176/deepRIV/ImputedTraits/code/adsp/FUSION/get_data.R')

# Loading the function we have defined in R.
stage1_r = robjects.globalenv['read_data']

file_num = int(sys.argv[-1])
pheno_study = sys.argv[1]
tissue = sys.argv[2]

stage1_gene_names = pd.read_csv('/home/panwei/he000176/deepRIV/ImputedTraits/results/adsp/gene/'+tissue+'/stage1_gene.csv')

# n1_samplesize = int(sys.argv[1])
runs = 70
start = (file_num-1)*runs
end = min(file_num*runs, stage1_gene_names.shape[0])

runs = end - start

# split_seeds = (ord('u'), ord('k'), ord('b'))
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

training_params = ({'learning_rate':0.00001,
                    'training_steps':500,
                    'decay_steps':150,
                    'batch_size':2048,
                    'decay_rate':1,
                    'patience':15},
                    {'learning_rate':0.00001,
                    'training_steps':500,
                    'batch_size':1024,
                    'decay_rate':1,
                    'patience':15},
                    {'learning_rate':0.00001,
                    'training_steps':500,
                    'batch_size':512,
                    'decay_rate':1,
                    'patience':15})

adsp_training_params = ({'learning_rate':0.00001,
                    'training_steps':500,
                    'decay_steps':150,
                    'decay_rate':1,
                    'patience':15},
                    {'learning_rate':0.000033,
                    'training_steps':500,
                    'decay_rate':.95,
                    'patience':15},
                    {'learning_rate':0.00033,
                    'training_steps':500,
                    'batch_size':32,
                    'decay_rate':.67,
                    'patience':15})

num_repeats = len(training_params)

data_path = '/home/panwei/he000176/deepRIV/ImputedTraits/ADSP/'
out_path = '/home/panwei/he000176/deepRIV/ImputedTraits/'
common_path = out_path + 'results/adsp/gene/'+ tissue +'/common_{}_{}.txt'.format(pheno_study,file_num)
IVa_path = out_path + 'results/adsp/gene/'+ tissue +'/IVa_{}_{}.txt'.format(pheno_study,file_num)
debug_path = out_path + 'results/debug/adsp/gene/'+ tissue +'/gene_{}_{}'.format(pheno_study,file_num)
stage1_path = out_path + 'results/adsp/gene/'+ tissue +'/stage1/'

COV = pd.read_csv(data_path + 'adsp_cov.csv')
ukb_cov_all = pd.read_csv('/home/panwei/he000176/deepRIV/UKB/data/covariates_age_baseline.txt',sep=' ')

logging.basicConfig(
    filename = debug_path + '.log',  # File to write logs to
    level = logging.DEBUG,  # Level of logging
    format = '%(asctime)s - %(levelname)s - %(message)s',  # Format of logs
    datefmt = '%Y-%m-%d %H:%M:%S'  # Date format in logs
)


temp_dirs = []

# gene_ids = pd.read_csv('/home/panwei/shared/GTEx_v8/GTEx_Analysis_v8_expression_EUR/expression_matrices/'+tissue+'.v8.EUR.normalized_expression.bed.gz',sep='\t').gene_id.values
gene_ids = stage1_gene_names.gene_names.values
ukb_igap = pd.read_csv('/home/panwei/he000176/deepRIV/ImputedTraits/ADSP/igap_imputed_ad.txt', sep = ' ')
ukb_eadb = pd.read_csv('/home/panwei/he000176/deepRIV/ImputedTraits/ADSP/eadb_imputed_ad.txt', sep = ' ')
ukb_proxy = pd.read_csv('/home/panwei/he000176/deepRIV/ImputedTraits/ADSP/igap_imputed_ad.txt', sep = ' ')
ukb_proxy.testtrue = ukb_proxy.testtrue - 1

try:
    for gene_ind in range(start, end):
        gene_ind = int(gene_ind)
        print(f"gene_ind: {gene_ind}")
        gene_name = gene_ids[gene_ind]
        rds_path = '/scratch.global/he000176/ImputedTraits/stage1_data/' + tissue + '/' + gene_name + '.rds'
        logging.info(f"Processing gene index: {gene_ind}")
        if not os.path.exists(rds_path):
            continue
        #converting it into r object for passing into r function
        with localconverter(robjects.default_converter + pandas2ri.converter):
            stage1_results = stage1_r(rds_path)
            if len(stage1_results.values())==0:
                logging.warning(f"Gene index {gene_ind} did not pass stage 1 screening")
                continue
            gene_name = robjects.conversion.rpy2py(stage1_results['gene'])
            gene_names = gene_name[0]
            #
            ukb_snp = robjects.conversion.rpy2py(stage1_results['ukb_snp_bed_reduced'])
            coefs_X1 = robjects.conversion.rpy2py(stage1_results['hatbetaX1'])
            coefs_X2 = robjects.conversion.rpy2py(stage1_results['hatbetaX2'])
            Y_ID = robjects.conversion.rpy2py(stage1_results['y_ukb_ID'])
            stage1_r2 = robjects.conversion.rpy2py(stage1_results['stage1_r2'])
            stage1_X2_r2 = robjects.conversion.rpy2py(stage1_results['stage1_X2_r2'])
            num_snps_original = robjects.conversion.rpy2py(stage1_results['num_snps_gtex'])
            num_snps_common = robjects.conversion.rpy2py(stage1_results['num_snps_common'])
            rs_id = robjects.conversion.rpy2py(stage1_results['rs_id'])
            adsp_snp = robjects.conversion.rpy2py(stage1_results['adsp_snp_bed_reduced'])
            y_adsp_ID = np.array(robjects.conversion.rpy2py(stage1_results['y_adsp_ID']))
            y_adsp = robjects.conversion.rpy2py(stage1_results['y_adsp_total'])
            num_individuals = Y_ID.shape[0]
            logging.debug(f"Gene {gene_names} results extracted")
            # if the test set is too small, skip
            if ukb_snp.shape[0] < 50000:
                print(2, gene_ind)
                continue
        
        if pheno_study=='igap':
            Y = pd.DataFrame({'id': ukb_igap.IID.values, 'pheno':ukb_igap.a40.values})
            Y = Y.set_index('id').loc[Y_ID].reset_index()
            logging.info("UKB pheno sample IDs align with ukb snp bed: {}".format(all(Y.id.values==Y_ID)))
            Y = Y.pheno.values
        elif pheno_study=='eadb':
            Y = pd.DataFrame({'id': ukb_eadb.IID.values, 'pheno':ukb_eadb.e40.values})
            Y = Y.set_index('id').loc[Y_ID].reset_index()
            logging.info("UKB pheno sample IDs align with ukb snp bed: {}".format(all(Y.id.values==Y_ID)))
            Y = Y.pheno.values
        elif pheno_study=='proxyAD':
            Y = pd.DataFrame({'id': ukb_proxy.IID.values, 'pheno':ukb_proxy.testtrue.values})
            Y = Y.set_index('id').loc[Y_ID].reset_index()
            logging.info("UKB pheno sample IDs align with ukb snp bed: {}".format(all(Y.id.values==Y_ID)))
            Y = Y.pheno.values
        
        if np.isnan(stage1_r2['lasso'][0]):
            stage1_r2['lasso'][0] = 0
        
        if np.isnan(stage1_r2['elastic_net'][0]):
            stage1_r2['elastic_net'][0] = 0
            
        if np.isnan(stage1_X2_r2['lasso'][0]):
            stage1_X2_r2['lasso'][0] = 0
        
        if np.isnan(stage1_X2_r2['elastic_net'][0]):
            stage1_X2_r2['elastic_net'][0] = 0
        
        if max(stage1_r2['lasso'][0], stage1_r2['elastic_net'][0]) < 0.01:
            logging.debug(f"{gene_name} did not pass stage 1 QC!")
            continue
        elif stage1_r2['lasso'][0] > stage1_r2['elastic_net'][0] and np.sum(coefs_X1.lasso!=0) > 0:
            hatbetaX1 = coefs_X1.lasso.values
        elif np.sum(coefs_X1.elastic_net!=0) > 0:
            hatbetaX1 = coefs_X1.elastic_net.values
        else:
            hatbetaX1 = np.zeros(coefs_X1.shape[0])
            logging.debug(f"{gene_name} did not pass stage 1 QC!")
            continue
            
        if stage1_X2_r2['lasso'][0] > stage1_X2_r2['elastic_net'][0] and np.sum(coefs_X2.lasso!=0) > 0:
            hatbetaX2 = coefs_X2.lasso.values
        elif np.sum(coefs_X2.elastic_net!=0) > 0:
            hatbetaX2 = coefs_X2.elastic_net.values
        else:
            hatbetaX2 = np.zeros(coefs_X2.shape[0])
        
              
        # convert large valus in y_adsp to nan
        mask = np.isin(adsp_snp, [0,1,2])
        adsp_snp = np.where(mask, adsp_snp, np.nan)    

        # aligh ukb_cov with ukb_ID
        ukb_cov = ukb_cov_all[ukb_cov_all['FID'].isin(Y_ID)]

        # align COV with data
        # Step 1: Filter COV to include only the rows where 'SampleID' is in y_adsp_ID.
        twas_cov = COV[COV['SampleID'].isin(y_adsp_ID)]

        # Create a mapping from ID to the index for y_adsp_ID.
        id_to_index = {id_: index for index, id_ in enumerate(y_adsp_ID)}

        # Step 2: Order X and y_adsp to match the filtered 'SampleID' order in COV.
        # Create a new order for the indices based on the order of 'SampleID' in filtered_COV.
        new_order = [id_to_index[id_] for id_ in twas_cov['SampleID']]

        # Reorder X and y_adsp according to the new order.
        adsp_snp = adsp_snp[new_order, :]
        y_adsp = y_adsp[new_order]
        y_adsp_ID = y_adsp_ID[new_order]

        logging.debug(f"ADSP Y aligned with covariates: {all(y_adsp_ID == twas_cov.SampleID.values)}")

        # scale covariates
        scaler_cov = StandardScaler().fit(twas_cov.iloc[:,1:])
        twas_cov_scaled = scaler_cov.transform(twas_cov.iloc[:,1:])
        twas_cov_scaled = pd.DataFrame(twas_cov_scaled, columns = twas_cov.columns[1:])

        # impute missing genotype 
        # imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
        imp_mean = SimpleImputer(strategy='most_frequent')
        ukb_snp_imputed = imp_mean.fit_transform(ukb_snp)
        adsp_snp_imputed = imp_mean.fit_transform(adsp_snp)

        scaler_ukb = StandardScaler().fit(ukb_snp_imputed)
        scaler_adsp = StandardScaler().fit(adsp_snp_imputed)

        ukb_X = scaler_ukb.transform(ukb_snp_imputed) @ hatbetaX1
        ukb_X2 = scaler_ukb.transform(ukb_snp_imputed) @ hatbetaX2 
        adsp_X = scaler_adsp.transform(adsp_snp_imputed) @ hatbetaX1
        adsp_X2 = scaler_adsp.transform(adsp_snp_imputed) @ hatbetaX2

        # regress out cov from y_adsp
        tmp_data = twas_cov_scaled.copy()
        tmp_data['y'] = y_adsp.copy()
        tmp_data['sex'] = pd.Categorical(tmp_data['sex'].astype(int))
        model = smf.ols('y ~ pc1 + pc2 + pc3 + pc4 + pc5 \
            + pc6 + pc7 + pc8 + pc9 + pc10 + pc11 + pc12 + pc13 \
            + pc14 + pc15 + pc16 + pc17 + pc18 + pc19 + pc20 + sex + age + age2 + sex:age + sex:age2', 
        data=tmp_data).fit()
        y_adsp_cov = model.resid.values

        logging.debug("Covariates regression completed")
        
        # adsp complete TWAS-L, TWAS-LQ
        pg_X1_adsp, pg_X2_adsp, pnl_X2_adsp, _ = TWAS(adsp_X, adsp_X2, y_adsp, twas_cov_scaled)
        # igap imputed ukb complete TWAS-L, TWAS-LQ
        pg_X1_ukb, pg_X2_ukb, pnl_X2_ukb, _ = TWAS(ukb_X, ukb_X2, Y)
        
        if np.sum(hatbetaX2)==0:
            pg_X2_adsp=pnl_X2_adsp=pg_X2_ukb=pnl_X2_ukb = 100

        common_results = pd.DataFrame({"gene_names":gene_names,
                            "pg_X1_adsp":pg_X1_adsp,
                            "pg_X2_adsp":pg_X2_adsp,
                            "pnl_X2_adsp":pnl_X2_adsp,
                            "pg_X1_ukb":pg_X1_ukb,
                            "pg_X2_ukb":pg_X2_ukb,
                            "pnl_X2_ukb":pnl_X2_ukb,
                            "stage1_r2":max(stage1_r2['lasso'][0], stage1_r2['elastic_net'][0]),
                            "stage1_X2_f2":max(stage1_X2_r2['lasso'][0], stage1_X2_r2['elastic_net'][0]),
                            "num_snps_original":num_snps_original[0],
                            "num_snps_common":num_snps_common[0],
                            "adsp_size":adsp_snp.shape[0],
                            "ukb_size":ukb_snp.shape[0],
                            "gene_ind":gene_ind
                            }, index = [0])
        common_results.to_csv(common_path, mode = 'a', index = False, 
                                sep = " ", header = not os.path.exists(common_path))
        
        logging.debug("TWAS-L & TWAS-LQ completed, results saved to " + common_path)

        IVa_pg_adsp = np.ones(num_repeats)
        IVa_pnl_adsp = np.ones(num_repeats)
        IVa_pg_ukb = np.ones(num_repeats)
        IVa_pnl_ukb = np.ones(num_repeats)
        mu_max = np.zeros(num_repeats)
        mu_min = np.zeros(num_repeats)
        for j in range(num_repeats):
            logging.info(f"DeLIVR training started. Repeat {j+1}/{num_repeats} for gene index {gene_ind}")
            # train test split
            data = utils.split_scale([adsp_snp], [y_adsp],hatbetaX1,0,['adsp'],
                cov_list = [twas_cov.iloc[:,1:]], **adsp_split_ratio[j])
            if data is None or np.unique(data['adsp']['train']['x']).shape[0] < 2:
                IVa_pg_adsp[j], IVa_pg_ukb = 100, 100
                IVa_pnl_adsp[j], IVa_pnl_ukb = 100, 100
                logging.warning(f"Insufficient training data for repeat {j+1} for protein index {gene_ind}")
                continue
            ##################### train and test on adsp #####################
            utils.regress_cov(data['adsp']) # regress out covariates
            logging.debug(f"Covariates regressed out for ADSP data")
            
            l2 = 0
            rsp_n1 = create_stage2((1,),1, l2 = l2)
            
            logging.debug(f"DeLIVR (ADSP) training started")
            
            _ = fit_stage2(rsp_n1, data['adsp']['train']['x'], data['adsp']['train']['y'], 
                data['adsp']['val']['x'], data['adsp']['val']['y'], **adsp_training_params[j])
            
            logging.debug(f"DeLIVR (ADSP) training finished. Testing started")
            
            # tests
            tmp = rsp_n1.predict(data['adsp']['test']['x']).squeeze()
            IVa_pg_adsp[j], _, IVa_pnl_adsp[j], _, _, _, _, _ = stage2Tests(data['adsp']['test']['x'], 
                data['adsp']['test']['y'], tmp)
            # save model
            # if IVa_pnl_n1[j] <= 0.05/100:
            if True:
                model_path = out_path + 'models/adsp/gene/' + tissue
                rsp_n1.save(model_path + '/adsp_{}_{}'.format(gene_ind, j))
            
            logging.debug(f"DeLIVR (ADSP) analysis finished. DeLIVR (LS-imp) analysis started.")
            
            ##################### now fit on ukb and test on adsp #####################
            logging.debug(f"Sample splitting for UKB")
            # train
            if pheno_study == 'proxyAD':
                logging.debug("Dimensions for ukb data: ukb_snp {}, Y {}, ukb_cov {}".format(ukb_snp.shape, Y.shape, ukb_cov.iloc[:,1:].shape))
                data = utils.split_scale([ukb_snp], [Y],hatbetaX1,0,['ukb'],
                    cov_list = [ukb_cov.iloc[:,1:]], **split_ratio[j])
                logging.debug(f"Covariates regression started for UKB proxy AD")
                utils.regress_cov(data['ukb'], keys = ['train','val']) # regress out covariates
                logging.debug(f"Covariates regressed out for UKB proxy AD")
            else:
                data = utils.split_scale([ukb_snp], [Y],hatbetaX1,0,['ukb'],
                                         **split_ratio[j])
            l2 = 0
            rsp_n1n2 = create_stage2((1,),1, l2 = l2)
            
            logging.debug(f"DeLIVR (LS-imp) training started")
            
            _ = fit_stage2(rsp_n1n2, data['ukb']['train']['x'], data['ukb']['train']['y'], 
                data['ukb']['val']['x'], data['ukb']['val']['y'], **training_params[j])
            
            logging.debug(f"DeLIVR (LS-imp) training finished. Testing started")
            
            # tests
            tmp = rsp_n1n2.predict(adsp_X).squeeze()
            IVa_pg_ukb[j], _, IVa_pnl_ukb[j], _, _, _, _, _ = stage2Tests(adsp_X, y_adsp_cov, tmp)
            # save model
            # if IVa_pnl_n1n2[j] <= 0.05/100:
            if True:
                model_path = out_path + 'models/adsp/gene/' + tissue
                rsp_n1n2.save(model_path + '/{}_{}_{}'.format(pheno_study,gene_ind, j))
            
            logging.debug(f"DeLIVR (LS-imp) analysis finished")

        gene_names = np.array(gene_names).reshape(-1,1)
        IVa_results = pd.DataFrame(gene_names)
        IVa_results = pd.concat([IVa_results, 
            pd.DataFrame(np.concatenate((IVa_pg_adsp, IVa_pnl_adsp,
                                        IVa_pg_ukb, IVa_pnl_ukb, 
                                        [y_adsp_ID.shape[0]], [Y_ID.shape[0]], [gene_ind+1])).reshape(1,-1))], 
            axis = 1)
        colnames = [['gene_names'], ['pg_adsp']*num_repeats, ['pnl_adsp']*num_repeats,
                    ['pg_ukb']*num_repeats, ['pnl_ukb']*num_repeats,
                    ['adsp_size'], ['ukb_size'], ['gene_ind']]
        IVa_results.columns = list(itertools.chain(*colnames))
        IVa_results.to_csv(IVa_path, mode = 'a', index = False, 
                                sep = " ", header = not os.path.exists(IVa_path))
        logging.debug(f"DeLIVR results (repeat {j+1}/{num_repeats}) successfully saved to " + IVa_path)
except Exception as e:
    logging.error(f"An error occurred: {e}")
    logging.error("Traceback:\n%s", traceback.format_exc())
    raise

