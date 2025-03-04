import sys
import pickle
import pdb
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as K
import statsmodels.api as sm
from tensorflow.keras import Model
from tensorflow.keras.layers import Activation, Add, Input, Dense, Dropout, Flatten
from tensorflow.keras.layers import BatchNormalization as BN
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import StratifiedShuffleSplit
from tensorflow.keras.layers import LeakyReLU, ReLU, Reshape, Concatenate
from tensorflow.keras.backend import expand_dims, squeeze, clear_session
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.constraints import max_norm
from sklearn.model_selection import train_test_split
from scipy.stats import norm


def create_stage1(X,y):
	design = sm.add_constant(X)
	trt1M = sm.OLS(y, design)
	trt1 = trt1M.fit()
	return trt1


def create_stage2(input_shape, 
	ouput_shape,
	layer_dims = [32,16,8,8,8],
	activation = tf.nn.relu,
	l2 = 0.05,
	use_bias = True,
	model = "original"):
	#
	if model=="original":
		input1 = Input(shape=input_shape)
		out = Dense(layer_dims[0], activation=activation, kernel_regularizer=K.regularizers.l2(l2=l2), 
					dtype = tf.float64, use_bias=use_bias)(input1)
		for i in range(1,len(layer_dims)):
			out = Dense(layer_dims[i], activation=activation, kernel_regularizer=K.regularizers.l2(l2=l2), 
						dtype = tf.float64, use_bias=use_bias)(out)
		#
		out = Dense(ouput_shape, activation=None, dtype = tf.float64, use_bias=True)(out)
		skp = Dense(ouput_shape, activation=None, dtype = tf.float64, use_bias=True)(input1)
		out = Add()([out,skp])
		rsp = Model(inputs=input1, outputs=out)
		return rsp
	elif model=="linear":
		input1 = Input(shape=input_shape)
		out = Dense(ouput_shape, activation=None, dtype = tf.float64, use_bias=True)(input1)
		rsp = Model(inputs=input1, outputs=out)
		return rsp
	elif model=="more_skip_layer":
		skps = {}
		input1 = Input(shape=input_shape)
		out = Dense(layer_dims[0], activation=activation, kernel_regularizer=K.regularizers.l2(l2=l2), 
					dtype = tf.float64, use_bias=use_bias)(input1)
		for i in range(1,len(layer_dims)):
			out = Dense(layer_dims[i], activation=activation, kernel_regularizer=K.regularizers.l2(l2=l2), 
						dtype = tf.float64, use_bias=use_bias)(out)
			if i < 3:
				skps['{}'.format(i)] = Dense(ouput_shape, activation=None, dtype = tf.float64, use_bias=True)(out)
		#
		out = Dense(ouput_shape, activation=None, dtype = tf.float64, use_bias=True)(out)
		skp = Dense(ouput_shape, activation=None, dtype = tf.float64, use_bias=True)(input1)
		out = Add()([out,skp])
		for i in range(1, 3):
			out = Add()([out, skps['{}'.format(i)]])
		rsp = Model(inputs=input1, outputs=out)
		return rsp



def fit_stage2(model, X, y, X_val, y_val,
	optimizer = "Adam", 
	loss = "MSE",
	learning_rate = 0.00033, 
	batch_size = 64, 
	training_steps = 8000,
	decay_steps = 200,
	decay_rate = 0.03,
	patience = 20,
	shuffle = True):
	#
	lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
					learning_rate,
					decay_steps=decay_steps,
					decay_rate=decay_rate,
					staircase=True)
	if optimizer=="Adam":
		opt = tf.keras.optimizers.Adam(learning_rate = lr_schedule)
	elif optimizer=="SGD":
		opt = tf.keras.optimizers.SGD(learning_rate = lr_schedule, momentum=0.9)

	if loss == "MSE":
		loss_f = tf.keras.losses.MSE
	model.compile(optimizer=opt,
					loss=loss_f,
					metrics=['MeanSquaredError'])
	erlystp = tf.keras.callbacks.EarlyStopping(monitor='val_mean_squared_error', min_delta = 0.00001, patience = patience)
	history = model.fit(X, y, batch_size=batch_size, epochs=training_steps, callbacks = [erlystp],
						validation_data=(X_val, y_val), shuffle = shuffle, verbose=1)
	return history


# def CV_l2(X,y,folds=5,seed=0,
# 	input_shape, 
# 	ouput_shape,
# 	layer_dims = [32,16,8,8,8],
# 	activation = tf.nn.relu,
# 	l2 = np.linspace(0,5,50),
# 	use_bias = True):
# 	kFold = StratifiedKFold(n_splits=folds, random_state=seed)
# 	for train, test in kFold.split(X, Y):
# 		model = create_stage2(input_shape, 
# 								ouput_shape,
# 								layer_dims,
# 								activation,
# 								l2 = 0.05,
# 								use_bias = True)
# 		 = train_test_split(X[train], y[train], test_size=0.2, random_state=seed)
# 		train_evaluate(model, X[train], Y[train], X[test], Y[test])


def tune_l2(X,y,
	input_shape, 
	ouput_shape,
	layer_dims = [32,16,8,8,8],
	activation = tf.nn.relu,
	log_l2 = np.linspace(-5,5,50),
	use_bias = True,
	seed=0):
	#
	x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
	x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=1) 
	#
	l2 = np.append(0,np.exp(log_l2))
	# l2 = np.exp(log_l2)
	test_losses = np.zeros(l2.shape[0])
	for i in range(l2.shape[0]):
		model = create_stage2(input_shape, 
							ouput_shape,
							layer_dims,
							activation,
							l2 = l2[i],
							use_bias = True)
		_ = fit_stage2(model, x_train, y_train, x_val, y_val)
		pred = model.predict(x_test).squeeze()
		test_losses[i] = np.sum((y_test-pred)**2)
	return l2[np.argmin(test_losses)]


def stage2Tests(x_test, y_test, pred):
	# global tests
	design = sm.add_constant(pred.reshape(-1,1))
	testM = sm.OLS(y_test, design)
	testM_results = testM.fit()
	p_global = testM_results.pvalues[-1]
	t_global = testM_results.tvalues[-1]
	linear_coef = testM_results.params[-1]
	linear_se = testM_results.bse[-1]
	#
	# Nonlinearity test
	testM = sm.OLS(y_test, sm.add_constant(x_test))
	testM_results = testM.fit()
	residuals = y_test - testM_results.predict(sm.add_constant(x_test)).reshape(y_test.shape)
	testM = sm.OLS(residuals, sm.add_constant(pred.reshape(-1,1)))
	testM_results = testM.fit()
	p_nonlinear = testM_results.pvalues[-1]
	t_nonlinear = testM_results.tvalues[-1]
	nonlinear_coef = testM_results.params[-1]
	nonlinear_se = testM_results.bse[-1]
	#
	return p_global, t_global, p_nonlinear, t_nonlinear, nonlinear_coef, nonlinear_se, linear_coef, linear_se

def cauchy_p(p_val, num_repeats):
	'''
		p_val: matrix of p-values, shape = (# samples, # repeats)
	'''
	cauchy_stat = np.tan((0.5-p_val)*np.pi).sum(axis = 1) / num_repeats
	P = 1/2 - np.arctan(cauchy_stat)/np.pi
	return P

def stage2Tests_C(x_test,y_test,pred,VarY_C):
	n = pred.shape[0]
	n_batches = n//VarY_C.shape[0]
	batch_size = VarY_C.shape[0]
	global_p_val = np.ones(n_batches)
	#nl_p_val = np.ones(n_batches)
	Var_alpha_all = np.zeros(n_batches)
	for i in range(n_batches):
		W_batch = pred[i*batch_size:(i+1)*batch_size]
		x_test_batch = x_test[i*batch_size:(i+1)*batch_size,:]
		x_test_batch = x_test_batch - x_test_batch.mean(axis=0)
		y_test_batch = y_test[i*batch_size:(i+1)*batch_size]
		y_test_batch = y_test_batch - y_test_batch.mean(axis=0)
		# get alpha
		_, _, _, _, nl_alpha, _, alpha, _ = stage2Tests(x_test_batch, 
			y_test_batch, W_batch)
		# global test
		Wa = W_batch * alpha
		Var_e = VarY_C - np.var(Wa) * np.eye(batch_size)
		tmp = 1/((W_batch**2).sum().squeeze())**2
		Var_alpha = tmp * W_batch.T @ Var_e @ W_batch
		global_p_val[i] = 1 - norm.cdf(alpha/np.sqrt(Var_alpha))
		Var_alpha_all[i] = Var_alpha
	combined_p = cauchy_p(global_p_val.reshape(1,-1), n_batches)
	return combined_p, np.mean(Var_alpha_all), np.std(Var_alpha_all)
	