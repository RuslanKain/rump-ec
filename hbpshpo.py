
#%%
from pyswarm import pso

from os import path
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import os

import numpy as np
import pandas as pd
import time
import random
from math import sqrt 
import random
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Cropping1D, Cropping2D, TimeDistributed, Dense, Dropout, LSTM, Conv1D, Flatten, Activation, MaxPooling1D
from json import loads
from scipy.ndimage.filters import uniform_filter1d
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow
import matplotlib.pyplot as plt 
import re

from skopt.space import Categorical, Real
from skopt.utils import use_named_args
from skopt import gp_minimize

from util import *
from tqdm import tqdm

#%%

def truncate(n, decimals=0):
    multiplier = 10 ** decimals
    return int(n * multiplier) / multiplier

def hasLSTM(model):
	firstlayer=model.layers[0].name[:4]
	if(firstlayer=='lstm'):
		answer=True
	else:
		answer=False
	return answer

def hasCONV1D(model):
	firstlayer=model.layers[0].name[:6]
	if(firstlayer=='conv1d'):
		answer=True
	else:
		answer=False
	return answer

def split_sequences(sequences, n_steps_in, n_steps_out):
	X, y = list(), list()
	for i in range(len(sequences)):
		end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out
		if out_end_ix > len(sequences):
			break
		seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix:out_end_ix, :]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)

def metrics(model, X_test, y_test_3d, scaler, cpu_user_col, cpu_system_col, cpu_idle_col, ram_col):#, upload_col, label_col):
  
  global lookahead

  
  results_dict = {'cpu_user_time_diff_observations':[], 'cpu_user_time_diff_predicted_observations':[], 'cpu_user_time_diff_mae':[], 'cpu_user_time_diff_rmse':[],
   'cpu_system_time_diff_observations':[], 'cpu_system_time_diff_predicted_observations':[],  'cpu_system_time_diff_mae':[], 'cpu_system_time_diff_rmse':[],
    'cpu_idle_time_diff_observations':[], 'cpu_idle_time_diff_predicted_observations':[],  'cpu_idle_time_diff_mae':[], 'cpu_idle_time_diff_rmse':[],
     'memory_observations':[], 'memory_predicted_observations':[], 'memory_mae':[], 'memory_rmse':[]}
     #,'net_sent_diff_observations':[], 'net_sent_diff_predicted_observations':[], 'net_sent_diff_mae':[], 'net_sent_diff_rmse':[]}
     #,'label_observations':[], 'label_predicted_observations':[], 'label_mae':[], 'label_rmse':[]}
  
  prediction_3d = model.predict(X_test)
  
  iter_list = [('cpu_user_time_diff',cpu_user_col), ('cpu_system_time_diff',cpu_system_col),('cpu_idle_time_diff',cpu_idle_col),('memory',ram_col)]#,('net_sent_diff',upload_col),('label',label_col)]

  for feature_name, feature_col in iter_list:
    print(feature_name)

    if model_type == 'with conv--':
      y_test_3d = y_test_3d.reshape(y_test_3d.shape[0],lookahead,X_test.shape[2])
      prediction_3d = prediction_3d.reshape(prediction_3d.shape[0],lookahead,X_test.shape[2])
    
    for row in tqdm(range(len(y_test_3d))):

      y_test=scaler.inverse_transform(y_test_3d[row,:,:])
      prediction=scaler.inverse_transform(prediction_3d[row,:,:])

      results_dict[feature_name+'_observations'].append(y_test[:,feature_col].tolist())
      prds = [0 if i < 0 else i for i in prediction[:,feature_col].tolist()]

      results_dict[feature_name+'_predicted_observations'].append(prds)
      results_dict[feature_name+'_rmse'].append(sqrt(mean_squared_error(y_test[:,feature_col], prds)))
      results_dict[feature_name+'_mae'].append(mean_absolute_error(y_test[:,feature_col], prds)) 

  


      
  
  return results_dict


def inference(model, X, pred_loc, pred_shift):
  #i=random.randrange(len(X))	
  startS=time.time()	
  prediction=model.predict(X[[pred_loc]])	
  timeS=time.time()-startS
  #i=random.randrange(len(X)-pred_shift)
  input = X[pred_loc:pred_loc+pred_shift+1]
  startB=time.time()
  batch_prediction=model.predict(input)
  timeB=time.time()-startB
  return timeS, timeB

def data_preparation(datasetfile, drop_list):
  dataframe = pd.read_csv(datasetfile, engine='python')
  
  if 'time_stamp' in dataframe.columns:
    dataframe = dataframe.drop(drop_list, axis=1)  
  
  cpu_user_column = dataframe.columns.get_loc("cpu_user_time_diff")
  cpu_system_column = dataframe.columns.get_loc("cpu_system_time_diff")
  cpu_idle_column = dataframe.columns.get_loc("cpu_idle_time_diff")
  ram_column = dataframe.columns.get_loc("memory")
  #upload_column = dataframe.columns.get_loc("net_sent_diff")
  #label_column = dataframe.columns.get_loc("label")

  dataframe = dataframe.fillna(0)
  dataset = dataframe.values
  dataset = np.nan_to_num(dataset)
  dataset = dataset.astype('float32')
  #scaler = MinMaxScaler(feature_range=(0, 1))
  scaler = MaxAbsScaler()
  scaler.fit(dataset)
  dataset = scaler.transform(dataset)
  return dataset, scaler, cpu_user_column, cpu_system_column, cpu_idle_column, ram_column#, upload_column, label_column

def remove_dupes(i, o):
	for j in range(len(i)):
		for k in range(len(i)):
			if (i[j]==i[k] and j!=k):
				if (o[j]>o[k]):
					i.pop(j)
					o.pop(j)
					return remove_dupes(i, o)
				else:
					i.pop(k)
					o.pop(k)
					return remove_dupes(i, o)
	return i, o

def train_test(model, dataset, lookback, lookahead, split, model_type):
  if hasLSTM(model) or hasCONV1D(model):
    dataset,datasetY=split_sequences(dataset,lookback,lookahead)
    if model_type == 'with lstm':
      datasetY=datasetY.reshape(datasetY.shape[0],datasetY.shape[1],datasetY.shape[2]) 
    elif model_type == 'with conv':
      datasetY=datasetY.reshape(datasetY.shape[0],datasetY.shape[2]*datasetY.shape[1]) 
  else: # not used 
    dataset,datasetY=split_sequences(dataset,1,1)  
    dataset=dataset.reshape(dataset.shape[0],dataset.shape[2])
    datasetY=datasetY.reshape(datasetY.shape[0],datasetY.shape[2])  
  train_size = int(len(dataset) * split)
  #test_size = len(dataset) - train_size
  trainX, testX = dataset[0:train_size,:], dataset[train_size:,:]
  trainY, testY = datasetY[0:train_size,:], datasetY[train_size:,:] 
  return trainX, testX, trainY, testY

def getOptimizer(x4, optimizer):
  learning_rate = x4
  if optimizer =='RMSprop':
    optimizer = tensorflow.keras.optimizers.RMSprop(learning_rate=learning_rate)
  elif optimizer =='Adam':
    optimizer = tensorflow.keras.optimizers.Adam(learning_rate=learning_rate)
  elif optimizer =='SGD':
    optimizer = tensorflow.keras.optimizers.SGD(learning_rate=learning_rate)
  elif optimizer =='Adagrad':
    optimizer = tensorflow.keras.optimizers.Adagrad(learning_rate=learning_rate)
  elif optimizer =='Adadelta':
    optimizer = tensorflow.keras.optimizers.Adadelta(learning_rate=learning_rate)
  elif optimizer =='Adamax':
    optimizer = tensorflow.keras.optimizers.Adamax(learning_rate=learning_rate)
  else:
    optimizer = tensorflow.keras.optimizers.Nadam(learning_rate=learning_rate)

  return optimizer

def build_model(activationFunction, optimizer):
  global dataset, defineModel, lookback, lookahead, neurons, numOfLayers, dropout, noOfLSTM_CONV, pool_size
  
  """
  if defineModel == False:
    defineModel = True
    models = ['with lstm', 'with conv']
    typeOfModel = random.choice(models)
  """
  typeOfModel = random.choice(['with lstm', 'with conv'])
  model = Sequential()
  neurons = int(round(x0))
  numOfLayers = int(round(x1))
  dropout = truncate(x3,1)
  lookback = int(round(x2))
  optimization = getOptimizer(x4, optimizer)
  noOfLSTM_CONV = int(round(x7))
  pool_size = int(round(x8))
  
  
 
  if typeOfModel == 'with lstm':
    for i in range(1,noOfLSTM_CONV+1):
        if i == 1:
            model.add(LSTM(neurons, input_shape=(lookback, dataset.shape[1]), activation=activationFunction, recurrent_activation='sigmoid', implementation=2, return_sequences=True))
        else:
            model.add(LSTM(neurons,activation=activationFunction, recurrent_activation='sigmoid', implementation=2, return_sequences=True)) 
  
    units = neurons
    for i in range(1,numOfLayers+1):
        units = int(round(units/2))
        if units > 0:    
          model.add(TimeDistributed(Dense(units, activation=activationFunction)))
          model.add(TimeDistributed(Dropout(dropout)))

    model.add(TimeDistributed(Dense(dataset.shape[1])))

    model.add(Cropping1D(cropping=(lookback-lookahead,0)))


  elif typeOfModel == 'with conv':
    for i in range(1,noOfLSTM_CONV+1):
        if i == 1:
            model.add(Conv1D(filters=neurons, kernel_size=2, padding='same', activation=activationFunction, input_shape=(lookback,dataset.shape[-1])))
            model.add(MaxPooling1D(pool_size=pool_size, padding='same')) 
        else:
            model.add(Conv1D(filters=neurons, kernel_size=2, padding='same', activation=activationFunction))
            model.add(MaxPooling1D(pool_size=pool_size, padding='same')) 
    model.add(Flatten())

    units = neurons
    for i in range(1,numOfLayers+1):
        units = int(round(units/2))
        if units > 0:    
          model.add(Dense(units, activation=activationFunction))
          model.add(Dropout(dropout))

    model.add(Dense(lookahead*dataset.shape[1]))

  model.compile(loss='mse',
                optimizer=optimization,
                metrics=['mae', 'mse'])
  return model, typeOfModel
  
def plot_pred_obs(pred_data, obs_data, title, unit, type, rpi, method, filter, save = False):
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.plot(pred_data,label = 'Predicted', color = 'red')
  ax.plot(obs_data,label = 'Observed', color = 'blue')
  ax.set_title(method +' - '+ title)#, color = 'white')
  ax.grid()
  ax.legend(loc='best')
  #ax.tick_params(axis='x', colors='white')
  #ax.tick_params(axis='y', colors='white')
  #ax.yaxis.label.set_color('white')
  #ax.xaxis.label.set_color('white')
  ax.set_xlabel('Datapoint')#, color = 'white')
  ax.set_ylabel(unit)#, color = 'white')
  if save == True:
    if filter == 1:
      f_path = 'Non Filtered'
    else:
      f_path = 'Filtered'
    plt_name = title.replace(" ", "_")
    try:
      os.makedirs(r"figures/{}/{}/{}/{} Step/{}".format(type,rpi,method,lookahead,f_path))
    except:
      pass
    save_path = r"figures/{}/{}/{}/{} Step/{}/{}_{}step_{}ma.png".format(type,rpi,method,lookahead,f_path,plt_name,lookahead,filter) 
    plt.savefig(save_path)
  else:
    plt.show()

#%%

global activationFunction, optimizer
dimensions  = [Categorical(['tanh','sigmoid','linear','relu'], name='activationFunction'), #Search space, all parameters are nomminal
              Categorical(['RMSprop', 'Adam', 'SGD', 'Adagrad', 'Adadelta', 'Adamax', 'Nadam'], name='optimizer')]
defact = ['tanh','sigmoid','linear','relu']
defopt =  ['RMSprop', 'Adam', 'SGD', 'Adagrad', 'Adadelta', 'Adamax', 'Nadam']
default_parameters = [random.choice(defact), random.choice(defopt)]

#%%

@use_named_args(dimensions=dimensions) #Combine Objective function with its search space
def gp_minimize_opt_function(activationFunction, optimizer):

  global loss, best_loss, dataset, lookback, trainings, rpi_name, model_name, lookahead, eval_split
  epochs = int(round(x5))
  batch_size = int(round(x6))
  graph = tensorflow.Graph()
  with tensorflow.compat.v1.Session(graph=graph):

    model, typeOfModel = build_model(activationFunction, optimizer)
    print('model type:', typeOfModel)
    print('model output layer:', model.layers[-1].output_shape)
    trainX, testX, trainY, testY = train_test(model, dataset, lookback, lookahead, eval_split, typeOfModel)
    model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, verbose=1, callbacks=[EarlyStopper]) 
    
    #if lookahead == 1:
    loss, _, _ = model.evaluate(testX, testY, verbose=2)

    var1 ="neurons = ",str(neurons)
    var2 = "layers = ",str(numOfLayers)
    var3 = "lookback = ",str(lookback)
    var4 ="dropout = ",str(dropout)
    var5 = "learning rate = ",str(x4)
    var6 = "epochs = ",str(epochs)
    var7 = "batch_size = ",str(batch_size)
    var8 = "number of lstm/conv1d layers = ",str(noOfLSTM_CONV)
    var9 = "number of pool size = ",str(pool_size)
    var10 = "activation function = ",activationFunction
    var11 = "optimizer = ",optimizer
    var12 = "model type = ",typeOfModel
    i = "--"+var1[0]+var1[1]+"||"+var2[0]+var2[1]+"||"+var3[0]+var3[1]+"||"+var4[0]+var4[1]+"||"+var5[0]+var5[1]+"||"+var6[0]+var6[1]+"||"+var7[0]+var7[1]+"||"+var8[0]+var8[1]+"||"+var9[0]+var9[1]+"||"+var10[0]+var10[1]+"||"+var11[0]+var11[1]+"||"+var12[0]+var12[1]+"--"   
    print(i)

    if loss < best_loss:
      best_loss = loss
      model.save('HBPSHPO_Models/{}/best_{}'.format(rpi_name, model_name))
      lb=testX.shape[1]
      print(lb, file=open('HBPSHPO_Models/{}/lookback_{}.txt'.format(rpi_name, model_name),'w'))
    
      with open("HBPSHPO_Models/{}/hyper_{}.txt".format(rpi_name,model_name), "w") as f:
        f.write(i)
        
      losses = loss
    
      #else:
      #  pass

  return loss

def pso_opt_fun(x):
  global bn, bayes_inputs, bayes_results, bayes_time, bt, median_loss, best_swarm_loss, counter
  global x0, x1, x2, x3, x4, x5, x6, x7, x8
  x0 = x[0]
  x1 = x[1]
  x2 = x[2]
  x3 = x[3]
  x4 = x[4]
  x5 = x[5]
  x6 = x[6]
  x7 = x[7]
  x8 = x[8]
  calls = 3
  
  bt=time.time()
  if not bayes_inputs:
    bayes = gp_minimize(func=gp_minimize_opt_function, dimensions=dimensions, acq_func='EI',n_calls=calls, n_random_starts=1, x0=default_parameters, model_queue_size=1)
  else:
    bayes = gp_minimize(func=gp_minimize_opt_function, dimensions=dimensions, acq_func='EI',n_calls=calls, n_random_starts=1, x0=bayes_inputs, y0=bayes_results, model_queue_size=1)
  bayes_time=bayes_time+time.time()-bt
  bn=bn+1	
  bayes_inputs.extend(bayes.x_iters)
  bayes_results.extend(bayes.func_vals) 		
  keep=calls*bn
  bayes_inputs=bayes_inputs[-keep:]
  bayes_results=bayes_results[-keep:]
  bayes_inputs, bayes_results=remove_dupes(bayes_inputs, bayes_results)
  error=min(bayes_results[-calls:])
  
  return error


#%%
#model_name = 'paper_data'
#mydatasetfile = 'https://docs.google.com/uc?export=download&id=1XhbneHtO6R5b2XD401J6kcfccMdl0c1f'

#%%

import warnings
warnings.filterwarnings('ignore', message='The objective has been evaluated at this point before.')

lookahead_list = [5,15]

rpi_name = 'RPi4B8GB_1800' #RPi4B8GB_1800, RPi4B4GB_1500, RPi4B2GB2_1500, RPi4B2GB1_1200
data_seq = 'pattern' #random, pattern
data_num = '' # _2, _3, _4

OLD_drop_lst = ['time_stamp','top_cpu_process','top_memory_process','running_cpu_process','label_running','label_top_cpu']
drop_lst =  ['time_stamp', 'link_quality', 'wifi_freq', 'link_quality_max', 'state', 'predicted', 'label']

#x0=neurons, x1=layers, x2=lookback
#x3=dropout, x4=learning rate, x5=epochs
#x6=batch_size, x7=number of lstm/conv1d layers, x8=pool_size
for lookahead in lookahead_list:
  
  print(f'Training for {lookahead} step prediction on {rpi_name}_{data_seq}{data_num}')
  EarlyStopper = EarlyStopping(patience=1, monitor='loss', mode='min')
  trainings=0
  best_loss = 1
  defineModel = False
  bn=0
  bayes_inputs = []
  bayes_results = []
  time_total=[]
  bayes_time_total=[]
  bt=0
  bayes_time=0

  model_name = f'{rpi_name}_{data_seq}_{lookahead*5}sec'
  

  lb = [1, 1, lookahead, 0.0, 0.001, 20, 1, 1, 1]
  ub = [512, 5, lookahead*2, 0.5, 0.2, 200, 1000, 5, 5]

  
  mydatasetfile = f"data/{rpi_name.split('_')[0]}/{rpi_name}MHz_res_usage_data_train_pred_rvp_{data_seq}_48hr{data_num}.csv"
  
  eval_split = 0.80
  dataset_name = mydatasetfile.split("/")[-1]
  dataset, scaler, cpu_user_column, cpu_system_column, cpu_idle_column, ram_column = data_preparation(mydatasetfile,drop_lst)
  dataframe = pd.read_csv(mydatasetfile, engine='python')
  dataframe = dataframe.drop(drop_lst, axis=1)
  dataframe = dataframe.fillna(0)

  training_flag = 1

  """first attempt"""
  
  try:
    start = time.time()

    xopt, fopt = pso(pso_opt_fun, lb, ub, swarmsize=10, omega=1, phip=0.5, phig=1.0, maxiter=10, minstep=2, minfunc=0.00015)
    mytime=time.time()-start
    print ("Best position"+str(xopt))
    print ("Loss:" + str(fopt))
    print('Training time:',mytime)
  except:
    print(f'{lookahead} steps finished for {rpi_name}')

#%%
lookahead_list = [1,2,5,10,15,30,60]
rpi_name = 'RPi4B8GB_1800' #RPi4B8GB_1800, RPi4B4GB_1500, RPi4B2GB2_1500, RPi4B2GB1_1200
data_seq = 'pattern' #pattern
data_num = '' # _2, _3, _4



OLD_drop_lst = ['time_stamp','top_cpu_process','top_memory_process','running_cpu_process','label_running','label_top_cpu']
drop_lst =  ['time_stamp', 'link_quality', 'wifi_freq', 'link_quality_max', 'state','label']
for i in lookahead_list:
    drop_lst.append(f'label - {i} step')
    drop_lst.append(f'predicted states - {i} step')
    drop_lst.append(f'log_likelihood - {i} step')

mydatasetfile = f"data/{rpi_name.split('_')[0]}/{rpi_name}MHz_res_usage_data_test_pred_rvp_{data_seq}_48hr{data_num}.csv"

#%%
lookahead_list = [5,15]

for lookahead in lookahead_list:
  print(f'testing {rpi_name} model for {lookahead} step lookahead')
  model_name = f'{rpi_name}_{data_seq}_{lookahead*5}sec'

  dataset_name = mydatasetfile.split("/")[-1]
  dataset, scaler, cpu_user_column, cpu_system_column, cpu_idle_column, ram_column = data_preparation(mydatasetfile,drop_lst)
  dataframe = pd.read_csv(mydatasetfile, engine='python')
  dataframe = dataframe.drop(drop_lst, axis=1)
  dataframe = dataframe.fillna(0)

  mymodel=keras.models.load_model('HBPSHPO_Models/{}/best_{}'.format(rpi_name, model_name))
  text_file = open("HBPSHPO_Models/{}/lookback_{}.txt".format(rpi_name, model_name), "r")
  hp_text_file = open("HBPSHPO_Models/{}/hyper_{}.txt".format(rpi_name, model_name), "r")
  lookback = int(text_file.read())
  model_type = hp_text_file.read(-1).split(' = ')[-1]

  hp_text_file.close()
  if hasLSTM(mymodel) or hasCONV1D(mymodel):
    dataset,datasetY=split_sequences(dataset,lookback,lookahead)
    if model_type == 'with lstm--':
      datasetY=datasetY.reshape(datasetY.shape[0],datasetY.shape[1],datasetY.shape[2]) 
    elif model_type == 'with conv--':
      datasetY=datasetY.reshape(datasetY.shape[0],datasetY.shape[1]*datasetY.shape[-1]) 
  else:
    dataset,datasetY=split_sequences(dataset,1,1)  
    dataset=dataset.reshape(dataset.shape[0],dataset.shape[2])
    datasetY=datasetY.reshape(datasetY.shape[0],datasetY.shape[2])  

  prediction_location = 0
  prediction_batch = 100

  obs_dict = metrics(mymodel, dataset, datasetY, scaler, cpu_user_column, cpu_system_column, cpu_idle_column, ram_column)#, upload_column, label_column)
  infS, infB = inference(mymodel, dataset, prediction_location, prediction_batch)


  mymodel.summary()
  file_o=open("HBPSHPO_Models/{}/hyper_{}.txt".format(rpi_name, model_name))   
  content=file_o.read()                 
  print(content)                       
  file_o.close()


  print('PSO results using', dataset_name, 'as dataset')

  print('Single inference time: %.3f s	batch inference time: %.3f s ' % (infS, infB))
  #if training_flag == 1:
    #print('Best mse: %.6f	mae: %.6f	rmse: %.6f	training time: %.0f s' % (mse, mae, sqrt(mse), mytime))
  #  print('DeepLearning,',mytime,',',',',infS,',',infB,',',sqrt(mse_cpu_user),',',mae_cpu_user,',',sqrt(mse_cpu_system),',',mae_cpu_system,',',sqrt(mse_cpu_idle),',',mae_cpu_idle,',',math.sqrt(mse_ram),',',mae_ram, file=open('results_{}.csv'.format(model_name),'a'))

  print('User CPU mae: %.6f User CPU rmse: %.6f' % (round(mean_absolute_error(obs_dict['cpu_user_time_diff_observations'],obs_dict['cpu_user_time_diff_predicted_observations']),3), round(sqrt(mean_squared_error(obs_dict['cpu_user_time_diff_observations'], obs_dict['cpu_user_time_diff_predicted_observations'])),3)))
  print('System CPU mae: %.6f System CPU rmse: %.6f' % (round(mean_absolute_error(obs_dict['cpu_system_time_diff_observations'],obs_dict['cpu_system_time_diff_predicted_observations']),3), round(sqrt(mean_squared_error(obs_dict['cpu_system_time_diff_observations'],obs_dict['cpu_system_time_diff_predicted_observations'])),3)))
  print('Idle CPU mae: %.6f Idle CPU rmse: %.6f' % (round(mean_absolute_error(obs_dict['cpu_idle_time_diff_observations'],obs_dict['cpu_idle_time_diff_predicted_observations']),3), round(sqrt(mean_squared_error(obs_dict['cpu_idle_time_diff_observations'],obs_dict['cpu_idle_time_diff_predicted_observations'])),3)))
  print('RAM mae: %.6f	RAM rmse: %.6f' % (round(mean_absolute_error(obs_dict['memory_observations'],obs_dict['memory_predicted_observations']),3), round(sqrt(mean_squared_error(obs_dict['memory_observations'],obs_dict['memory_predicted_observations'])),3)))
  #print('Upload mae: %.6f	Upload rmse: %.6f' % (round(mean_absolute_error(obs_dict['net_sent_diff_observations'],obs_dict['net_sent_diff_predicted_observations']),3), round(sqrt(mean_squared_error(obs_dict['net_sent_diff_observations'],obs_dict['net_sent_diff_predicted_observations'])),3)))
  #print('label mae: %.6f	label rmse: %.6f' % (round(mean_absolute_error(obs_dict['label_observations'],obs_dict['label_predicted_observations']),3), round(sqrt(mean_squared_error(obs_dict['label_observations'],obs_dict['label_predicted_observations'])),3)))

  save = {'User CPU mae':[round(mean_absolute_error(obs_dict['cpu_user_time_diff_observations'],obs_dict['cpu_user_time_diff_predicted_observations']),3)], 'User CPU rmse':[round(sqrt(mean_squared_error(obs_dict['cpu_user_time_diff_observations'],obs_dict['cpu_user_time_diff_predicted_observations'])),3)],
          'System CPU mae':[round(mean_absolute_error(obs_dict['cpu_system_time_diff_observations'],obs_dict['cpu_system_time_diff_predicted_observations']),3)], 'System CPU rmse':[round(sqrt(mean_squared_error(obs_dict['cpu_system_time_diff_observations'],obs_dict['cpu_system_time_diff_predicted_observations'])),3)],
          'Idle CPU mae':[round(mean_absolute_error(obs_dict['cpu_idle_time_diff_observations'],obs_dict['cpu_idle_time_diff_predicted_observations']),3)], 'Idle CPU rmse':[round(sqrt(mean_squared_error(obs_dict['cpu_idle_time_diff_observations'],obs_dict['cpu_idle_time_diff_predicted_observations'])),3)],
          'RAM mae':[round(mean_absolute_error(obs_dict['memory_observations'],obs_dict['memory_predicted_observations']),3)], 'RAM rmse':[round(sqrt(mean_squared_error(obs_dict['memory_observations'],obs_dict['memory_predicted_observations'])),3)]}
          #,'Upload mae':[round(mean_absolute_error(obs_dict['net_sent_diff_observations'],obs_dict['net_sent_diff_predicted_observations']),3)], 'Upload rmse':[round(sqrt(mean_squared_error(obs_dict['net_sent_diff_observations'],obs_dict['net_sent_diff_predicted_observations'])),3)]}
          #,'label mae':[round(mean_absolute_error(obs_dict['label_observations'],obs_dict['label_predicted_observations']),3)], 'label rmse':[round(sqrt(mean_squared_error(obs_dict['label_observations'],obs_dict['label_predicted_observations'])),3)]}

  pd.DataFrame(save).to_csv(r"Results/{}/HBPSHPO_Error_Results_{}.csv".format(rpi_name.split('_')[0],model_name))
  pd.DataFrame(obs_dict).to_csv(r"Results/{}/HBPSHPO_Results_{}.csv".format(rpi_name.split('_')[0],model_name))


# %%
"""Get prediction figures"""

rpi_name = 'RPi4B2GB2_1500' #RPi4B8GB_1800, RPi4B4GB_1500, RPi4B2GB2_1500, RPi4B2GB1_1200
data_seq = 'random' # random, pattern
data_num = '' # _2, _3, _4
method = 'HBPSHPO' # HBPSHPO, HSMM
lookahead_list = []
filter = 25 # 1, 25, 50, 100

for lookahead in lookahead_list:
  print(f"{method} figures for {rpi_name}_{data_seq}{data_num} for {lookahead} step")
  model_name = f'{rpi_name}_{data_seq}_{lookahead*5}sec'
  if method == 'HSMM':
    obs_dict = pd.read_csv(f"Results/{rpi_name.split('_')[0]}/HSMM_Results_rvp_{data_seq}_48hr{data_num}_1500sec_lb_{lookahead*5}sec_pw.csv")
  elif method == 'HBPSHPO' == 1:
    obs_dict = pd.read_csv(f"Results/{rpi_name.split('_')[0]}/HBPSHPO_Results_{model_name}.csv")

  user_cpu_obs, user_cpu_pred_obs = [], []
  system_cpu_obs, system_cpu_pred_obs = [], []
  idle_cpu_obs, idle_cpu_pred_obs = [], []
  ram_obs, ram_pred_obs = [], []
  #label_obs, label_pred_obs = [], []
  #net_sent_obs, net_sent_pred_obs = [], []
  

  for i in range(0,len(obs_dict),lookahead):
    if True:
      user_cpu_obs.extend(loads(obs_dict['cpu_user_time_diff_observations'][i]))
      user_cpu_pred_obs.extend(loads(obs_dict['cpu_user_time_diff_predicted_observations'][i]))
      system_cpu_obs.extend(loads(obs_dict['cpu_system_time_diff_observations'][i]))
      system_cpu_pred_obs.extend(loads(obs_dict['cpu_system_time_diff_predicted_observations'][i]))
      idle_cpu_obs.extend(loads(obs_dict['cpu_idle_time_diff_observations'][i]))
      idle_cpu_pred_obs.extend(loads(obs_dict['cpu_idle_time_diff_predicted_observations'][i]))
      #net_sent_obs.extend(loads(obs_dict['net_sent_diff_observations'][i]))
      #net_sent_pred_obs.extend(loads(obs_dict['net_sent_diff_predicted_observations'][i]))
      ram_obs.extend(loads(obs_dict['memory_observations'][i]))
      ram_pred_obs.extend(loads(obs_dict['memory_predicted_observations'][i]))
      #label_obs.extend(loads(obs_dict['label_observations'][i]))
      #label_pred_obs.extend(loads(obs_dict['label_predicted_observations'][i]))
    else:
      user_cpu_obs.extend(obs_dict['cpu_user_time_diff_observations'][i])
      user_cpu_pred_obs.extend(obs_dict['cpu_user_time_diff_predicted_observations'][i])
      system_cpu_obs.extend(obs_dict['cpu_system_time_diff_observations'][i])
      system_cpu_pred_obs.extend(obs_dict['cpu_system_time_diff_predicted_observations'][i])
      idle_cpu_obs.extend(obs_dict['cpu_idle_time_diff_observations'][i])
      idle_cpu_pred_obs.extend(obs_dict['cpu_idle_time_diff_predicted_observations'][i])
      #net_sent_obs.extend(obs_dict['net_sent_diff_observations'][i])
      #net_sent_pred_obs.extend(obs_dict['net_sent_diff_predicted_observations'][i])
      ram_obs.extend(obs_dict['memory_observations'][i])
      ram_pred_obs.extend(obs_dict['memory_predicted_observations'][i])
      #label_obs.extend(obs_dict['label_observations'][i])
      #label_pred_obs.extend(obs_dict['label_predicted_observations'][i])

  plot_pred_obs(uniform_filter1d(user_cpu_pred_obs, size=filter),uniform_filter1d(user_cpu_obs, size=filter) , 'CPU User Time', 'Seconds', data_seq + data_num, rpi_name.split('_')[0], method, filter, False)
  plot_pred_obs(uniform_filter1d(system_cpu_pred_obs, size=filter), uniform_filter1d(system_cpu_obs, size=filter) , 'CPU System Time', 'Seconds', data_seq + data_num, rpi_name.split('_')[0], method, filter, False)
  plot_pred_obs(uniform_filter1d(idle_cpu_pred_obs, size=filter), uniform_filter1d(idle_cpu_obs, size=filter), 'CPU Idle Time', 'Seconds', data_seq + data_num, rpi_name.split('_')[0], method, filter, False)
  plot_pred_obs(uniform_filter1d(ram_pred_obs, size=filter), uniform_filter1d(ram_obs, size=filter), 'RAM', 'Memory (%)', data_seq + data_num, rpi_name.split('_')[0], method, filter, False)
  #plot_pred_obs(uniform_filter1d(net_sent_pred_obs, size=filter) , uniform_filter1d(net_sent_obs, size=filter), 'Network Upload', 'Bytes', method, filter, True)

# %%
plt.rcParams.update({'font.size': 20})

def mk_groups(data):
  try:
      newdata = data.items()
  except:
      return

  thisgroup = []
  groups = []
  for key, value in newdata:
      newgroups = mk_groups(value)
      if newgroups is None:
          thisgroup.append((key, value))
      else:
          thisgroup.append((key, len(newgroups[-1])))
          if groups:
              groups = [g + n for n, g in zip(newgroups, groups)]
          else:
              groups = newgroups
  return [thisgroup] + groups

def add_line(ax, xpos, ypos):
  line = plt.Line2D([xpos, xpos], [ypos + .1, ypos],
                    transform=ax.transAxes, color='black')
  line.set_clip_on(False)
  ax.add_line(line)

def label_group_bar(ax, mae_data, rmse_data):
  
  mae_groups, rmse_groups = mk_groups(mae_data), mk_groups(rmse_data)
  mae_xy, rmse_xy = mae_groups.pop(), rmse_groups.pop()
  mae_x, mae_y = zip(*mae_xy)
  rmse_x, rmse_y = zip(*rmse_xy)
  mae_ly, rmse_ly = len(mae_y), len(rmse_y)
  mae_xticks, rmse_xticks = np.arange(1, mae_ly + 1), np.arange(1, mae_ly + 1)


  mae_bars = ax.bar(mae_xticks, mae_y, align='center',color='green', label = 'MAE')
  rmse_bars = ax.bar(rmse_xticks, rmse_y, bottom = mae_y, align='center',color='orange', label = "RMSE")

  hatches = ['','/']*7

  for i in range(len(mae_bars)):
    mae_bars[i].set(hatch = hatches[i])
    rmse_bars[i].set(hatch = hatches[i])

  ax.set_xticks(mae_xticks)
  #ax.set_xticklabels(mae_x)
  ax.set_xlim(.5, mae_ly + .5)
  ax.yaxis.grid(True)

  scale = 1. / mae_ly
  for pos in range(mae_ly + 1):
      add_line(ax, pos * scale, -.1)
  ypos = -.2
  while mae_groups:
      group = mae_groups.pop()
      pos = 0
      for label, rpos in group:
          lxpos = (pos + .5 * rpos) * scale
          if 'step' in label:
            ax.text(lxpos, ypos, label, ha='center', transform=ax.transAxes)
          else:
            ax.text(lxpos, ypos, label, ha='center', transform=ax.transAxes, rotation=90)
          add_line(ax, pos * scale, ypos)
          pos += rpos
      add_line(ax, pos * scale, ypos)
      ypos -= .1

# %%
"""Get error figures"""

rpi_name = 'RPi4B2GB2_1500' #RPi4B8GB_1800, RPi4B4GB_1500, RPi4B2GB2_1500, RPi4B2GB1_1200
data_seq = 'random' # random, pattern
data_num = '' # _2, _3, _4

lookahead_list = [1,2,5,10,15,30,60]
HSMM_error_list, HBPSHPO_error_list = [], []


HSMM_columns = ['cpu_user_time_diff_mae','cpu_user_time_diff_rmse','cpu_system_time_diff_mae','cpu_system_time_diff_rmse','cpu_idle_time_diff_mae','cpu_idle_time_diff_rmse','memory_mae','memory_rmse']
HBPSHPO_columns = ['User CPU mae','User CPU rmse','System CPU mae','System CPU rmse','Idle CPU mae','Idle CPU rmse','RAM mae','RAM rmse']
Resources = {'User CPU Time':['cpu_user_time_diff_mae','cpu_user_time_diff_rmse','User CPU mae','User CPU rmse'],
             'System CPU Time':['cpu_system_time_diff_mae','cpu_system_time_diff_rmse','System CPU mae','System CPU rmse'],
             'Idle CPU Time':['cpu_idle_time_diff_mae','cpu_idle_time_diff_rmse', 'Idle CPU mae','Idle CPU rmse'],
             'Memory %':['memory_mae','memory_rmse','RAM mae','RAM rmse']}

for res in Resources:
  mae_data_dict, rmse_data_dict = {}, {}
  for lookahead in lookahead_list:

    mae_data_dict[f"{lookahead} step"] = {}
    rmse_data_dict[f"{lookahead} step"] = {}


    print(f"Error figures for {rpi_name}_{data_seq}{data_num} for {lookahead} step")

    model_name = f'{rpi_name}_{data_seq}_{lookahead*5}sec'

    HSMM_temp = pd.read_csv(f"Results/{rpi_name.split('_')[0]}/HSMM_Error_Results_rvp_{data_seq}_48hr{data_num}_1500sec_lb_{lookahead*5}sec_pw.csv")
    HBPSHPO_temp = pd.read_csv(f"Results/{rpi_name.split('_')[0]}/HBPSHPO_Error_Results_{model_name}.csv")

    mae_data_dict[f"{lookahead} step"]['HSMM'] = HSMM_temp[Resources[res][0]]
    mae_data_dict[f"{lookahead} step"]['HBPSHPO'] = HBPSHPO_temp[Resources[res][2]]
    rmse_data_dict[f"{lookahead} step"]['HSMM'] = HSMM_temp[Resources[res][1]]
    rmse_data_dict[f"{lookahead} step"]['HBPSHPO'] = HBPSHPO_temp[Resources[res][3]]

  fig = plt.figure(figsize=(14,14))
  ax = fig.add_subplot(1,1,1)
  ax.set_title(res)
  label_group_bar(ax, mae_data_dict, rmse_data_dict)
  plt.legend(loc='best')
  fig.subplots_adjust(bottom=0.3)
  fig.savefig(r"figures/{}/{}/{} Error Plot.png".format(data_seq + data_num,rpi_name.split('_')[0], res))
  plt.show()
# %%
