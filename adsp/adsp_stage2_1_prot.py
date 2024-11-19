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
r['source']('/home/panwei/he000176/deepRIV/ImputedTraits/code/adsp/adsp_stage1_FUN_impute_prot.R')

# Loading the function we have defined in R.
stage1_r = robjects.globalenv['stage1']

file_num = int(sys.argv[-1])
pheno_study = sys.argv[1]
# n1_samplesize = int(sys.argv[1])
runs = 20
start = (file_num-1)*runs + 1
end = min(file_num*runs + 1, 19697)
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
common_path = out_path + 'results/adsp/prot/common_{}_{}.txt'.format(pheno_study,file_num)
IVa_path = out_path + 'results/adsp/prot/IVa_{}_{}.txt'.format(pheno_study,file_num)
debug_path = out_path + 'results/debug/adsp/prot/prot_{}_{}'.format(pheno_study,file_num)

COV = pd.read_csv(data_path + 'adsp_cov.csv')
ukb_cov_all = pd.read_csv('/home/panwei/he000176/deepRIV/UKB/data/covariates_age_baseline.txt',sep=' ')

logging.basicConfig(
    filename = debug_path + '.log',  # File to write logs to
    level = logging.DEBUG,  # Level of logging
    format = '%(asctime)s - %(levelname)s - %(message)s',  # Format of logs
    datefmt = '%Y-%m-%d %H:%M:%S'  # Date format in logs
)


# hdl = pd.read_csv(data_path + 'igap_imputed_ad.txt', dtype = np.float32)
# hdl.columns = ['id', 'hdl']
# labels = hdl.hdl.values
# labels_imputed = pd.read_csv(data_path + 'batchsize20000/yhat.txt', 
#     dtype = np.float32, sep = ' ', header = None).values.squeeze()
# VarY_C = np.loadtxt('/home/panwei/he000176/deepRIV/ImputedTraits/data/boot_varEY/Var_Y.txt')

temp_dirs = []

try:
    for gene_ind in range(start, end):
        gene_ind = int(gene_ind)
        logging.info(f"Processing protein index: {gene_ind}")
        #converting it into r object for passing into r function
        with localconverter(robjects.default_converter + pandas2ri.converter):
            stage1_results = stage1_r(gene_ind,pheno_study)
            if len(stage1_results.values())==0:
                logging.warning(f"Protein index {gene_ind} did not pass stage 1 screening")
                continue
            gene_name = robjects.conversion.rpy2py(stage1_results['protein_id'])
            gene_names = gene_name[0]
            #
            ukb_snp = robjects.conversion.rpy2py(stage1_results['ukb_snp_beds2'])
            ukb_prot_sample_size = robjects.conversion.rpy2py(stage1_results['ukb_protein_sample_size'])
            # ukb_cov = robjects.conversion.rpy2py(stage1_results['cov_s2'])
            intercept = robjects.conversion.rpy2py(stage1_results['intercept'])
            hatbetaX1 = robjects.conversion.rpy2py(stage1_results['hatbetaX1'])
            intercept2 = robjects.conversion.rpy2py(stage1_results['intercept2'])
            hatbetaX2 = robjects.conversion.rpy2py(stage1_results['hatbetaX2'])
            Y_ID = robjects.conversion.rpy2py(stage1_results['y_ukb_ID'])
            stage1_fstat = robjects.conversion.rpy2py(stage1_results['stage1_fstat'])
            stage1_pval = robjects.conversion.rpy2py(stage1_results['stage1_pval'])
            stage1_X2_fstat = robjects.conversion.rpy2py(stage1_results['stage1_X2_fstat'])
            num_snps = robjects.conversion.rpy2py(stage1_results['num_snps'])
            stage1_sigma = robjects.conversion.rpy2py(stage1_results['stage1_sigma'])
            Y = robjects.conversion.rpy2py(stage1_results['y_ukb_total'])
            rs_id = robjects.conversion.rpy2py(stage1_results['rs_id'])
            adsp_snp = robjects.conversion.rpy2py(stage1_results['adsp_snp_bed'])
            y_adsp_ID = np.array(robjects.conversion.rpy2py(stage1_results['y_adsp_ID']))
            y_adsp = robjects.conversion.rpy2py(stage1_results['y_adsp_total'])
            tmp_dir_list = robjects.conversion.rpy2py(stage1_results['tmp_dir_list'])
            temp_dirs = [str(dir) for dir in tmp_dir_list]
            num_individuals = Y_ID.shape[0]
            logging.debug(f"Protein {gene_names} results extracted")
            # if the test set is too small, skip
            if ukb_snp.shape[0] < 50000:
                logging.warning(f"Insufficient SNPs for protein index {gene_ind}")
                continue

        # convert large valus in y_adsp to nan
        mask = np.isin(adsp_snp, [0,1,2])
        adsp_snp = np.where(mask, adsp_snp, np.nan)

        # aligh ukb_cov with ukb_ID
        ukb_cov = ukb_cov_all[ukb_cov_all['FID'].isin(Y_ID)]

        logging.debug("ukb_cov shape: {}, ukb outcome shape: {}".format(ukb_cov.shape, Y.shape))
        
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

        ukb_X = scaler_ukb.transform(ukb_snp_imputed) @ hatbetaX1 + intercept
        ukb_X2 = scaler_ukb.transform(ukb_snp_imputed) @ hatbetaX2 + intercept2
        adsp_X = scaler_adsp.transform(adsp_snp_imputed) @ hatbetaX1 + intercept
        adsp_X2 = scaler_adsp.transform(adsp_snp_imputed) @ hatbetaX2 + intercept2

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

        common_results = pd.DataFrame({"gene_names":gene_names,
                            "pg_X1_adsp":pg_X1_adsp,
                            "pg_X2_adsp":pg_X2_adsp,
                            "pnl_X2_adsp":pnl_X2_adsp,
                            "pg_X1_ukb":pg_X1_ukb,
                            "pg_X2_ukb":pg_X2_ukb,
                            "pnl_X2_ukb":pnl_X2_ukb,
                            "stage1_fstat":stage1_fstat,
                            "stage1_X2_fstat":stage1_X2_fstat,
                            "num_snps":num_snps,
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
            logging.info(f"DeLIVR training started. Repeat {j+1}/{num_repeats} for protein index {gene_ind}")
            # train test split
            logging.debug(f"Sample splitting for ADSP")
            data = utils.split_scale([adsp_snp], [y_adsp],hatbetaX1,intercept,['adsp'],
                cov_list = [twas_cov.iloc[:,1:]], **adsp_split_ratio[j])
            if data is None or np.unique(data['adsp']['train']['x']).shape[0] < 2:
                IVa_pg_adsp[j], IVa_pg_ukb = 100, 100
                IVa_pnl_adsp[j], IVa_pnl_ukb = 100, 100
                logging.warning(f"Insufficient training data for repeat {j+1} for protein index {gene_ind}")
                continue
            ##################### train and test on adsp #####################
            logging.debug(f"Covariates regression started for ADSP")
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
            if False:
                model_path = out_path + 'models/hdl/combine_p/'
                rsp_n1.save(model_path + 'n1_{}_{}'.format(gene_ind, j))
            
            logging.debug(f"DeLIVR (ADSP) analysis finished. DeLIVR (LS-imp) analysis started.")

            ##################### now fit on ukb and test on adsp #####################
            # train
            logging.debug(f"Sample splitting for UKB")
            if pheno_study == 'proxyAD':
                logging.debug("Dimensions for ukb data: ukb_snp {}, Y {}, ukb_cov {}".format(ukb_snp.shape, Y.shape, ukb_cov.iloc[:,1:].shape))
                data = utils.split_scale([ukb_snp], [Y],hatbetaX1,intercept,['ukb'],
                    cov_list = [ukb_cov.iloc[:,1:]], **split_ratio[j], debug=True)
                        
                logging.debug(f"Covariates regression started for UKB proxy AD")
                utils.regress_cov(data['ukb'], keys = ['train','val']) # regress out covariates
                logging.debug(f"Covariates regressed out for UKB proxy AD")
            else:
                data = utils.split_scale([ukb_snp], [Y],hatbetaX1,intercept,['ukb'],
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
            if False:
                model_path = out_path + 'models/hdl/combine_p/'
                rsp_n1n2.save(model_path + 'n1n2_{}_{}'.format(gene_ind, j))
                
            logging.debug(f"DeLIVR (LS-imp) analysis finished")


        gene_names = np.array(gene_names).reshape(-1,1)
        IVa_results = pd.DataFrame(gene_names)
        IVa_results = pd.concat([IVa_results, 
            pd.DataFrame(np.concatenate((IVa_pg_adsp, IVa_pnl_adsp,
                                        IVa_pg_ukb, IVa_pnl_ukb, 
                                        [y_adsp_ID.shape[0]], [Y_ID.shape[0]], [ukb_prot_sample_size[0]], [gene_ind])).reshape(1,-1))], 
            axis = 1)
        colnames = [['gene_names'], ['pg_adsp']*num_repeats, ['pnl_adsp']*num_repeats,
                    ['pg_ukb']*num_repeats, ['pnl_ukb']*num_repeats,
                    ['adsp_size'], ['ukb_size'], ['ukb_prot_sample_size'],['gene_ind']]
        IVa_results.columns = list(itertools.chain(*colnames))
        IVa_results.to_csv(IVa_path, mode = 'a', index = False, 
                                sep = " ", header = not os.path.exists(IVa_path))
        logging.debug(f"DeLIVR results (repeat {j+1}/{num_repeats}) successfully saved to " + IVa_path)
except Exception as e:
    logging.error(f"An error occurred: {e}")
    logging.error("Traceback:\n%s", traceback.format_exc())
    raise


# def cleanup():
#     def remove_directory(dir_path):
#         if os.path.exists(dir_path):
#             try:
#                 shutil.rmtree(dir_path)
#                 print(f"Successfully removed directory: {dir_path}")
#             except Exception as e:
#                 print(f"Failed to remove directory {dir_path}: {e}")
#         else:
#             print(f"Directory does not exist: {dir_path}")
    
#     for dir in temp_dirs:
#         remove_directory(dir)

# Register cleanup function to be called on exit
# atexit.register(cleanup)

# Call the .Last function in R without quitting the R session
# robjects.r(".Last()")