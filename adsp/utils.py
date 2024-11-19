import os
import h5py
import pickle
import pdb
import scipy
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import statsmodels.api as sm
import statsmodels.formula.api as smf
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects import numpy2ri
from rpy2.robjects.conversion import localconverter
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
# from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from random import choices, shuffle
from collections import Counter


class SimpleImputer:
    """
    Impute missing values in a dataset with the specified strategy.
    """
    def __init__(self, strategy='mean'):
        if strategy not in ['mean', 'median', 'most_frequent']:
            raise ValueError("Invalid strategy. Supported strategies are 'mean', 'median', 'most_frequent'.")
        self.strategy = strategy
        self.statistics_ = None

    def fit(self, X, y=None):
        """
        Fit the imputer on X.

        Parameters:
        X (array-like): The input data with missing values (numpy array or pandas DataFrame).
        y: Ignored.

        Returns:
        self: Returns the instance itself.
        """
        X = pd.DataFrame(X)

        if self.strategy == 'mean':
            self.statistics_ = X.mean()
        elif self.strategy == 'median':
            self.statistics_ = X.median()
        elif self.strategy == 'most_frequent':
            self.statistics_ = X.mode().iloc[0]

        return self

    def transform(self, X):
        """
        Impute all missing values in X.

        Parameters:
        X (array-like): The input data with missing values (numpy array or pandas DataFrame).

        Returns:
        array-like: The data with imputed values.
        """
        if self.statistics_ is None:
            raise ValueError("The SimpleImputer instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")

        X = pd.DataFrame(X)
        X_imputed = X.fillna(self.statistics_)
        
        return X_imputed.values

    def fit_transform(self, X, y=None):
        """
        Fit to data, then transform it.

        Parameters:
        X (array-like): The input data with missing values (numpy array or pandas DataFrame).
        y: Ignored.

        Returns:
        array-like: The data with imputed values.
        """
        return self.fit(X, y).transform(X)


def shuffle_data(x, y, z):
    tmp = list(zip(*x, y, z))
    shuffle(tmp)
    shuffled = list(zip(*tmp))
    x_shuffled = [np.array(xi) for xi in shuffled[:-2]]
    y_shuffled = np.array(shuffled[-2])
    z_shuffled = np.array(shuffled[-1])
    return x_shuffled, y_shuffled, z_shuffled


def split_scale(z_list, y_list, hatbetaX1, intercept, data_source_keys, 
    cov_list=None, 
    test_ratio=0.45, 
    val_ratio=0.1, 
    random_state=0,
    debug = False):
    
    if debug:
        logging.basicConfig(
            filename = '/home/panwei/he000176/deepRIV/ImputedTraits/results/debug/adsp/utils.log',  # File to write logs to
            level = logging.DEBUG,  # Level of logging
            format = '%(asctime)s - %(levelname)s - %(message)s',  # Format of logs
            datefmt = '%Y-%m-%d %H:%M:%S'  # Date format in logs
        )
    
    if len(data_source_keys) != len(z_list):
        raise ValueError("The number of data source keys must match the number of data sources.")
    
    results = {}
    for idx, (key, z, y) in enumerate(zip(data_source_keys, z_list, y_list)):
        cov = None if cov_list is None else cov_list[idx]
        
        # Splitting into training, validation, and test sets
        # Handle test_ratio == 0 case
        if test_ratio == 0:
            if cov is None:
                if debug:
                    ########## DEBUG ##########
                    logging.debug("Data shapes: z: {}, y: {}, cov: {}".format(z.shape, y.shape, cov.shape))
                    ###########################
                z_train, z_val, y_train, y_val = train_test_split(z, y, test_size=val_ratio, random_state=random_state)
            else:
                if debug:
                    ########## DEBUG ##########
                    logging.info("split with covariates, test ratio is 0")
                    logging.debug("Data shapes: z: {}, y: {}, cov: {}".format(z.shape, y.shape, cov.shape))
                    ###########################
                z_train, z_val, y_train, y_val, cov_train, cov_val =\
                 train_test_split(z, y, cov, test_size=val_ratio, random_state=random_state)
            z_test_scaled, x_test_scaled, y_test_scaled, cov_test =\
            (None, None, None, None)  # No test data
        else:
            # Standard splitting process
            if cov is None:
                z_train, z_test, y_train, y_test =\
                 train_test_split(z, y, test_size=test_ratio, random_state=random_state)
            else:
                z_train, z_test, y_train, y_test, cov_train, cov_test =\
                 train_test_split(z, y, cov, test_size=test_ratio, random_state=random_state)

            if cov is None:
                z_train, z_val, y_train, y_val =\
                 train_test_split(z_train, y_train, test_size=val_ratio / (1 - test_ratio), random_state=random_state)
            else:
                z_train, z_val, y_train, y_val, cov_train, cov_val =\
                 train_test_split(z_train, y_train, cov_train, test_size=val_ratio / (1 - test_ratio), random_state=random_state)

        if debug:
            ########## DEBUG ##########
            logging.debug(f"Data shapes: z_train: {z_train.shape}, z_val: {z_val.shape}, z_test: {z_test.shape if test_ratio != 0 else 'N/A'}")
            logging.debug(f"Before imputation - z_train mean: {np.mean(z_train, axis=0)}, std: {np.std(z_train, axis=0)}")
            ###########################
                
        # Impute and scale z
        # imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
        imp_mean = SimpleImputer(strategy='most_frequent')
        z_train = imp_mean.fit_transform(z_train)
        z_val = imp_mean.transform(z_val)
        
        if debug:
            ########## DEBUG ##########
            logging.debug(f"After imputation - z_train mean: {np.mean(z_train, axis=0)}, std: {np.std(z_train, axis=0)}")
            ###########################

        scaler_z = StandardScaler().fit(z_train)
        z_train_scaled = scaler_z.transform(z_train)
        z_val_scaled = scaler_z.transform(z_val)
        
        if debug:
            ########## DEBUG ##########
            logging.debug(f"After scaling - z_train_scaled mean: {np.mean(z_train_scaled, axis=0)}, std: {np.std(z_train_scaled, axis=0)}")
            ###########################

        # Calculate x and scale x
        x_train = z_train_scaled @ hatbetaX1 + intercept
        x_val = z_val_scaled @ hatbetaX1 + intercept
        
        if debug:
            ########## DEBUG ##########
            logging.debug(f"x_train values - mean: {np.mean(x_train)}, std: {np.std(x_train)}")
            ###########################

        scaler_x = StandardScaler().fit(x_train.reshape(-1,1))
        x_train_scaled = scaler_x.transform(x_train.reshape(-1,1))
        x_val_scaled = scaler_x.transform(x_val.reshape(-1,1))

        if debug:
            ########## DEBUG ##########
            logging.debug(f"x_train_scaled stats - mean: {np.mean(x_train_scaled)}, std: {np.std(x_train_scaled)}")
            ###########################

        # Scale y
        scaler_y = StandardScaler().fit(y_train.reshape(-1, 1))
        y_train_scaled = scaler_y.transform(y_train.reshape(-1, 1))
        y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1))

        # Scale cov
        if cov is not None:
            scaler_cov = StandardScaler().fit(cov_train)
            cov_train_scaled = pd.DataFrame(scaler_cov.transform(cov_train), 
                columns = cov.columns)
            cov_val_scaled = pd.DataFrame(scaler_cov.transform(cov_val),
                columns = cov.columns)

        if test_ratio != 0:
            z_test = imp_mean.transform(z_test)
            z_test_scaled = scaler_z.transform(z_test)

            x_test = z_test_scaled @ hatbetaX1 + intercept
            x_test_scaled = scaler_x.transform(x_test.reshape(-1,1))

            y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1))

            if cov is not None:
                cov_test_scaled = pd.DataFrame(scaler_cov.transform(cov_test),
                    columns = cov.columns)

        # Pack results into a dictionary for each data source
        data_dict = {"train":{
                    "x":x_train_scaled,
                    "y":y_train_scaled,
                    "z":z_train_scaled
                },
                "val":{
                    "x":x_val_scaled, 
                    "y":y_val_scaled,
                    "z":z_val_scaled
                },
                "test":{
                    "x":x_test_scaled,
                    "y":y_test_scaled,
                    "z":z_test_scaled
                },
            }
        # data_dict = {
        #     # 'z_train': z_train_scaled, 'z_val': z_val_scaled, 'z_test': z_test_scaled,
        #     'mean_x_train': x_train_scaled, 'mean_x_val': x_val_scaled, 'mean_x_test': x_test_scaled,
        #     'y_train': y_train_scaled, 'y_val': y_val_scaled, 'y_test': y_test_scaled,
        # }
        if cov is not None:
            data_dict["train"].update({"cov": cov_train_scaled})
            data_dict["val"].update({"cov":cov_val_scaled})
            if test_ratio != 0:
                data_dict["test"].update({"cov":cov_test_scaled})
        results[key] = data_dict

    return results

def regress_cov(data, keys = ['train','val','test']):
    # Convert 'sex' to a categorical variable with 2 levels
    for key in keys:
        data[key]['cov']['sex'] = \
            pd.Categorical(data[key]['cov']['sex'].astype(int))
    #
    # Fit the model on the training set
    train_data = data['train']['cov'].copy()
    train_data['y'] = data['train']['y'].copy()
    model = smf.ols('y ~ pc1 + pc2 + pc3 + pc4 + pc5 \
        + pc6 + pc7 + pc8 + pc9 + pc10 + pc11 + pc12 + pc13 \
        + pc14 + pc15 + pc16 + pc17 + pc18 + pc19 + pc20 + sex + age + age2 + sex:age + sex:age2', 
        data=train_data).fit()
    # Calculate residuals for train, val, and test sets
    for key in keys:
        y_pred = model.predict(data[key]['cov'])
        data[key]['y'] = data[key]['y'] - y_pred.to_numpy().reshape(data[key]['y'].shape)
