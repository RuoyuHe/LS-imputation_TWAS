import pandas as pd, numpy as np 
import sys
import statsmodels.api as sm
import os 
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from LSimp import imputeY


file_num = int(sys.argv[-1])

data_path = '/home/panwei/he000176/deepRIV/ImputedTraits/data/for_revision/'
gene_ind = np.loadtxt('/home/panwei/he000176/deepRIV/ImputedTraits/data/gene_ind.txt', dtype = np.int32)
hdl = pd.read_csv('/home/panwei/shared/LSimputedHDLtrait/original_hdl.csv', dtype = np.float32)

effect_snp = pd.read_csv(data_path + '/effect_snps.txt', sep=' ', low_memory=True)
effect_snp_id = pd.read_csv(data_path + '/effect_snps_ID.txt', sep=' ', header = None)
total_snp = effect_snp.values

# get basic_Y
basic_beta = np.random.normal(loc=0.1, scale=np.sqrt(0.05), size=total_snp.shape[1])
basic_Y = total_snp @ basic_beta + np.random.normal(loc=0, scale=np.sqrt(0.007), size=total_snp.shape[0])
np.savetxt(data_path + f'rep_{file_num}/effect_beta.txt' , basic_beta)


Y = basic_Y.copy()
typeI_Y = basic_Y.copy()
g = gene_ind[0]
ID_path = data_path + 'effect_snps_ID.txt'
ref_ID = np.loadtxt(ID_path, dtype = np.int32)
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
gamma = 0.007


# np.random.seed(0)

f = lambda x: np.where(x < -0.5, 0.5 * x, np.where(x <= 0.5, 2 * x, 0.5 * x))

for g in gene_ind:
    print(g)
    snp_path = data_path + f'../{g}_bed.txt'
    hatbeta_path = data_path + f'../{g}_hatbetaX1.txt'
    ID_path = data_path + f'../{g}_ID.txt'
    if not(os.path.exists(snp_path) and os.path.exists(hatbeta_path) and os.path.exists(ID_path)):
        continue
    ukb_snp = pd.read_csv(snp_path)
    ukb_snp = imp_mean.fit_transform(ukb_snp)
    # hatbetaX1 = pd.read_csv(hatbeta_path).values.squeeze()
    #
    ID = np.loadtxt(ID_path, dtype = np.int32)
    #
    print('changed ID order: ', g)
    idx = np.array([np.where(ID == ref_ID[i])[0][0] for i in range(ref_ID.shape[0])])
    ukb_snp = ukb_snp[idx]
    #
#     if total_snp.shape[0]==0:
#         total_snp = ukb_snp.copy()
#     else:
#         total_snp = np.concatenate((total_snp, ukb_snp), axis = 1)
    total_snp = np.concatenate((total_snp, ukb_snp), axis = 1)
    e1 = np.random.normal(loc=0, scale=np.sqrt(gamma), size=total_snp.shape[0])
    e2 = np.random.normal(loc=0, scale=np.sqrt(gamma), size=total_snp.shape[0])
    e3 = np.random.normal(loc=0, scale=np.sqrt(gamma), size=total_snp.shape[0])
    U = np.random.normal(loc=0, scale=np.sqrt(gamma), size=total_snp.shape[0])
    #
    hatbetaX1 = np.random.normal(loc=0.1, scale=np.sqrt(0.1), size=ukb_snp.shape[1])
    X = StandardScaler().fit_transform(ukb_snp) @ hatbetaX1 + U + e1
    Y = Y + f(X) + U + e2
    typeI_Y = typeI_Y + U + e3
    #
    X_path = data_path + f'rep_{file_num}/{g}_simX.txt'
    np.savetxt(X_path, X)

Y_path = data_path + f'rep_{file_num}/Y.txt'
np.savetxt(Y_path, Y)
typeI_Y_path = data_path + f'rep_{file_num}/typeI_Y.txt'
np.savetxt(typeI_Y_path, typeI_Y)

gwas_size = 137162

gwas_idx = np.random.choice(np.arange(ref_ID.shape[0]), size = (gwas_size,), replace = False)
gwas_idx.sort()
gwas_idx_path = data_path + f'rep_{file_num}/gwas_idx.txt'
np.savetxt(gwas_idx_path, gwas_idx)

simulation_idx = set(np.arange(ref_ID.shape[0])).difference(set(gwas_idx))
simulation_idx = np.array(list(simulation_idx))
simulation_idx.sort()
simulation_idx_path = data_path + f'rep_{file_num}/simulation_idx.txt'
np.savetxt(simulation_idx_path, simulation_idx)

gwas_beta_train = []
gwas_betaSE_train = []
for snp in range(total_snp.shape[1]):
    design = sm.add_constant((total_snp[gwas_idx,snp] - total_snp[gwas_idx,snp].mean(axis=0)).reshape(-1,1))
    m = sm.OLS(Y[gwas_idx], design)
    m_results = m.fit()
    gwas_beta_train.append(m_results.params[-1])
    gwas_betaSE_train.append(m_results.bse[-1])

gwas_beta_train = np.array(gwas_beta_train)
gwas_betaSE_train = np.array(gwas_betaSE_train)
gwas_train_path = data_path + f'rep_{file_num}/gwas_beta_train.csv'
gwas_train = pd.DataFrame({'beta':gwas_beta_train, 'SE':gwas_betaSE_train})
gwas_train.to_csv(gwas_train_path, index = False)
y_n2, t = imputeY(total_snp[simulation_idx,:], gwas_beta_train)
y_n2_path = data_path + f'rep_{file_num}/imputed_Y.txt'
np.savetxt(y_n2_path, y_n2)

# for typeI Y
gwas_beta = []
gwas_betaSE = []
for snp in range(total_snp.shape[1]):
    design = sm.add_constant((total_snp[gwas_idx,snp] - total_snp[gwas_idx,snp].mean(axis=0)).reshape(-1,1))
    m = sm.OLS(typeI_Y[gwas_idx], design)
    m_results = m.fit()
    gwas_beta.append(m_results.params[-1])
    gwas_betaSE.append(m_results.bse[-1])

gwas_beta = np.array(gwas_beta)
gwaS_betaSE = np.array(gwas_betaSE)
gwas_typeIY = pd.DataFrame({'beta':gwas_beta, 'SE':gwaS_betaSE})
gwas_typeIY.to_csv(data_path + f'rep_{file_num}/gwas_beta_typeIY.csv', index = False)

y_n2, t = imputeY(total_snp[simulation_idx,:], gwas_beta)
y_n2_path = data_path + f'rep_{file_num}/imputed_typeIY.txt'
np.savetxt(y_n2_path, y_n2)


#
# tmp_idx = np.random.choice(np.arange(simulation_idx.shape[0]), size = gwas_idx.shape, replace = False)
tmp_idx = np.arange(simulation_idx.shape[0])
gwas_beta_impute_test = []
gwas_betaSE_impute_test = []
for snp in range(total_snp.shape[1]):
	design = sm.add_constant((total_snp[simulation_idx[tmp_idx],snp] - total_snp[simulation_idx[tmp_idx],snp].mean(axis=0)).reshape(-1,1))
	m = sm.OLS(y_n2[tmp_idx], design)
	m_results = m.fit()
	gwas_beta_impute_test.append(m_results.params[-1])
	gwas_betaSE_impute_test.append(m_results.bse[-1])

gwas_beta_impute_test = np.array(gwas_beta_impute_test)
gwas_betaSE_impute_test = np.array(gwas_betaSE_impute_test)
gwas_impute_test_path = data_path + f'rep_{file_num}/gwas_beta_impute_test.csv'
gwas_impute_test = pd.DataFrame({'beta':gwas_beta_impute_test, 'SE':gwas_betaSE_impute_test})
gwas_impute_test.to_csv(gwas_impute_test_path, index = False)


# gwas test
gwas_beta_test = []
gwas_betaSE_test = []
for snp in range(total_snp.shape[1]):
    design = sm.add_constant((total_snp[simulation_idx[tmp_idx],snp] - total_snp[simulation_idx[tmp_idx],snp].mean(axis=0)).reshape(-1,1))
    m = sm.OLS(Y[simulation_idx[tmp_idx]], design)
    m_results = m.fit()
    gwas_beta_test.append(m_results.params[-1])
    gwas_betaSE_test.append(m_results.bse[-1])

gwas_beta_test = np.array(gwas_beta_test)
gwas_betaSE_test = np.array(gwas_betaSE_test)
gwas_test_path = data_path + f'rep_{file_num}/gwas_beta_test.csv'
gwas_test = pd.DataFrame({'beta':gwas_beta_test, 'SE':gwas_betaSE_test})
gwas_test.to_csv(gwas_test_path, index = False)

# impute train gwas
# y_n2_train, t = imputeY(total_snp[gwas_idx,:], gwas_beta_train)
# print(y_n2_train.shape)
# gwas_beta_impute_train = []
# gwas_betaSE_impute_train = []
# for snp in range(total_snp.shape[1]):
# 	design = sm.add_constant((total_snp[gwas_idx,snp] - total_snp[gwas_idx,snp].mean(axis=0)).reshape(-1,1))
# 	print(design.shape)
# 	m = sm.OLS(y_n2_train, design)
# 	m_results = m.fit()
# 	gwas_beta_impute_train.append(m_results.params[-1])
# 	gwas_betaSE_impute_train.append(m_results.bse[-1])

# gwas_beta_impute_train = np.array(gwas_beta_impute_train)
# gwas_betaSE_impute_train = np.array(gwas_betaSE_impute_train)
# gwas_impute_train_path = data_path + f'rep_{file_num}/gwas_beta_impute_train.csv'
# gwas_impute_train = pd.DataFrame({'beta':gwas_beta_impute_train, 'SE':gwas_betaSE_impute_train})
# gwas_impute_train.to_csv(gwas_impute_train_path, index = False)