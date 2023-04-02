#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from ordered_set import OrderedSet
import copy 
from statistics import mean
from sklearn.preprocessing import normalize
import pickle
from scipy.stats import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
from tqdm import tqdm
from statistics import stdev
from sklearn.metrics import accuracy_score, multilabel_confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from json import loads
from comparison_test import Comparator
import os
import time
from scipy.signal import savgol_filter
#%%
import pyhsmm
import pyhsmm.internals
import pyhsmm.basic.distributions as distributions

#%% 
class HDP_HSMM():

    def __init__(self):
        self.COMP = Comparator()
    
    def _df_filter(self, df, column, savgol_window_size = 5, savgol_degree = 2, sma_window_size = 2, ema_window_size = 1):

        ts = df[column]
            
        #Savistky-Golay Filter
        ts_sg = pd.DataFrame(savgol_filter(ts, savgol_window_size, savgol_degree))
        ts_sg[ts_sg < 0] = 0 
        df[column+f'_sg_{savgol_window_size}_{savgol_degree}'] = ts_sg

        #Simple Moving Average Filter
        ts_sma = ts.rolling(sma_window_size, min_periods=1).mean()
        df[column+f'_sma_{sma_window_size}'] = ts_sma

        #Exponential Moving Average Filter
        ts_ema = ts.ewm(ema_window_size).mean()
        df[column+f'_ema_{ema_window_size}'] = ts_ema 

        return df

 
    def df_diff_and_filter(self, df, columns):
            """Places the difference betweeen to rows of dataframe in a new column named with '_diff' """
            
            df_columns = df.columns

            for col in columns: 
                if col in df_columns:

                    if col != 'memory':
                        df[col+'_diff'] = df[col].diff()
                        df = self._df_filter(df, col+'_diff')
                    else:
                        df = self._df_filter(df, col)
        
                else:
                    for column in df_columns:
                        if col in column and column+'_diff' not in df.columns:
                            if column != 'memory':
                                df[column+'_diff'] = df[column].diff()
                                df = self._df_filter(df, column+'_diff')
                            else:
                                df = self._df_filter(df, column)
            
            return df

    def transition_matrix(transitions):
        n = 1+ max(transitions) #number of states

        M = [[0]*n for _ in range(n)]

        for (i,j) in zip(transitions,transitions[1:]):
            M[i][j] += 1

        #now convert to probabilities:
        for row in M:
            s = sum(row)
            if s > 0:
                row[:] = [f/s for f in row]
        return M

    
    def save_model(self, model, device_name, model_name, filter, feature_names, fig = 'save'):

        if filter =='':
            filter = 'NoFilter'

        os.makedirs(os.path.join(r'HSMM_Models', device_name.split('_')[0], filter), exist_ok=True)
        with open('HSMM_Models/{}/{}/{}_{}_{}.pickle'.format(device_name.split('_')[0],filter,device_name,feature_names,model_name),'wb') as outfile:
            pickle.dump(model,outfile,protocol=-1)
            
        fig = plt.figure()
                
        plt.clf()
        model.plot()
        #model.plot_observations()
        #model.plot_stateseq()
        plt.gcf().suptitle('HDP-HSMM for {}_{}_{}_{}'.format(device_name,feature_names,model_name,filter)) 
        plt.tight_layout()
        
        
        if fig == 'save':
            plt.savefig('figures/{}_{}_{}_{}.png'.format(device_name,feature_names,model_name,filter))
        else:
            plt.show()


    def run_HSMM(self, data, features, extra_states = 0, model_count = 4, kappa = 0.05, progprint_xrange_var = 400):


        true_labels= data['state']
        
        states = list(OrderedSet(true_labels)) 
        
        data = data.drop(data.columns[[0,-1]], axis=1)
        
        Nmax = len(states) + extra_states
        
        data = data.reset_index(drop=True)

        #data = normalize(data[['cpu_user_time_diff','cpu_system_time_diff','cpu_idle_time_diff','memory','net_sent_diff']])
        data = data[features].to_numpy() 
        
        obs_dim = len(data[0])
        
        obs_hypparams = {'mu_0':np.zeros(obs_dim),
                'sigma_0':np.eye(obs_dim),
                'kappa_0': kappa,
                'nu_0':obs_dim+10}
                
        dur_hypparams = {'alpha_0':2*10,
                         'beta_0':2}

        distributions.DurationDistribution

        obs_distns = [distributions.Gaussian(**obs_hypparams) for state in range(Nmax)]
        dur_distns = [distributions.PoissonDuration(**dur_hypparams) for state in range(Nmax)]

        posteriormodel = pyhsmm.models.WeakLimitHDPHSMM(
                alpha=6.,gamma=6., # better to sample over these; see concentration-resampling.py
                init_state_concentration=6., # pretty inconsequential
                obs_distns=obs_distns,
                dur_distns=dur_distns)

        posteriormodel.add_data(data)

        models = []
                
        for idx in pyhsmm.pyhsmm.util.text.progprint_xrange(progprint_xrange_var):
            posteriormodel.resample_model()
            if (idx+1) % int(progprint_xrange_var/model_count) == 0:
                models.append(copy.deepcopy(posteriormodel))
                
        model = models[-1]
        return model, model.stateseqs, true_labels, states
    
    
    def get_HSMM_state_seq(self, data, model_path, device_name, model_count = 4, progprint_xrange_var = 800, plot = False):
        
        objects = []
        true_labels = data['state']
        
        with (open(model_path, "rb")) as openfile:
            while True:
                try:
                    objects.append(pickle.load(openfile))
                except EOFError:
                    break

        model = objects[0]
            
        if plot == True:
            model.plot()
            plt.gcf().suptitle('HDP-HSMM for {}'.format(device_name))
            plt.tight_layout()
            plt.show()
            
        return objects[0], objects[0].stateseqs, true_labels
        
    def HSMM_pred(self, model, seed_start, seed_end, pred_window):
        
        global df
        
        obs, stateseq = model.predict(df[seed_start:seed_end],pred_window)
        log_likelihood = model.log_likelihood(obs)

        return obs, stateseq, log_likelihood
    
    def metrics_plots(self, obs, pred_obs, features):# pred_stateseq, labels_running, labels_top_cpu, features):
               
                      
        """
        plt.plot(real_stateseq[0], color = 'red', label = 'states')
        plt.plot(pred_stateseq, color = 'blue', label = 'predicted')
        plt.legend(loc='best')
        plt.grid()
        plt.show()
        """
        obs_dict = {feature:[] for feature in features}
        pred_obs_dict = {feature:[] for feature in features}
        
        results_dict = {}
        
        for pred_ob in pred_obs:
            for feature in features:
                pred_obs_dict[feature].append(pred_ob[features.index(feature)])
                
        for ob in obs:
            for feature in features:
                obs_dict[feature].append(ob[features.index(feature)])
                
        for feature in features:

            results_dict[feature+'_observations'] = obs_dict[feature]
            results_dict[feature+'_predicted_observations'] = pred_obs_dict[feature]
            
            results_dict[feature+'_rmse'] = sqrt(mean_squared_error(obs_dict[feature],pred_obs_dict[feature]))
            results_dict[feature+'_mae'] = mean_absolute_error(obs_dict[feature],pred_obs_dict[feature])


            """state_sequences
            plt.plot(data[feature][test_start_idx:], color = 'red', label = 'obs')
            plt.plot(pred_obs_dict[feature][test_start_idx:], color = 'blue', label = 'predicted')
            plt.title(feature)
            plt.legend(loc='best')
            plt.grid()
            plt.show()
            """
            
        return results_dict 
        
    def merge_datasets(self, save, df_path, dataset_list):

        df = pd.concat(dataset_list)
        
        if save == True:
            df.to_csv(df_path)
            
        return df
    
    def merge_dataset(self, device_name,dataset_name_list, new_name):
        
        prep_data_list = list()
        num_list = ''
        
        for dataset_name in dataset_name_list:
            dataset_path = '~/git_repos/u-worc/data/{}/{}_res_usage_data_{}.csv'.format(device_name.split('_')[0],device_name,dataset_name)
            prep_data_list.append(pd.read_csv(dataset_path, index_col = 'time_stamp'))
        
        merge_dataset_path = '~/git_repos/u-worc/data/{}/{}_res_usage_data_{}.csv'.format(device_name.split('_')[0],device_name, new_name)
        return self.merge_datasets(True, merge_dataset_path, prep_data_list)

    def preprocess_data(self, device_name, freq, data_name, features, test = False):
        """Preprocessing labeled data"""

        #Read Data, create difference feature, and clean nans
        data_path = f"~/git_repos/u-worc/data/{device_name.split('_')[0]}/{device_name}_{freq}MHz_res_usage_data_{data_name}.csv"
        labeled_data = pd.read_csv(data_path, index_col = 'time_stamp')
        labeled_data = SM.df_diff_and_filter(labeled_data, features)
        labeled_data = labeled_data.fillna(0)


        #Move label column to the end
        labeled_data_cols = labeled_data.columns.tolist()
        oldindex = labeled_data_cols.index('state')
        labeled_data_cols.insert(len(labeled_data_cols), labeled_data_cols.pop(oldindex))
        labeled_data = labeled_data[labeled_data_cols]
        
        # remove first row due to diff = 0
        labeled_data = labeled_data.iloc[1: , :] 
        # remove rows with tranisition saving data states
        labeled_data = labeled_data[labeled_data['state'] != 'transition']
        labeled_data = labeled_data[labeled_data['state'] != 'saving data']

        if test == True:
            # For quick test
            labeled_data = labeled_data[:2000] #for testing only

        # train/test split        
        labeled_data_train = labeled_data[:int(0.7*len(labeled_data))]
        labeled_data_test = labeled_data[int(0.7*len(labeled_data)):]
        

        return labeled_data_train, labeled_data_test

    def check_duplicate_label(self, dict):
        
        for key1, val1 in dict.items():

            for key2, val2 in dict.items():
                if key1 != key2:
                    if val2 == val1:
                        print('Duplicate labels detected, not saving model')
                        return False
                        
                    else:
                        pass
        return True
        
    def save_label_accuracies(self, labels, device_name, model_name, filter, feature_names):
        
        if filter =='':
            filter = 'NoFilter'

        os.makedirs(os.path.join(r'HSMM_Models', device_name.split('_')[0], filter), exist_ok=True)
        labels.to_csv(os.path.join(r'HSMM_Models', device_name.split('_')[0], filter, f'{device_name}_{model_name}_label_accuracies.csv'))

    
    def grid_search(self, labeled_data_train, device_name, model_name, filter, features,feature_names, extras, kappas, iters, save='save'):
        
        max_accuracy = 0
        best_extra, best_kappa, best_iter = 0, 0, 0
        model_saved = False
        for extra in extras:
            for kap in kappas:
                for iter in iters:

                    print(f'Training for {extra} extra states, kappa = {kap}, and iters = {iter}')

                    
                    temp_model, temp_statesseqs, temp_true_labels, temp_states = self.run_HSMM(labeled_data_train, features, extra_states = extra, kappa = kap, progprint_xrange_var =iter) 
                    
                    labeled_data_train['predicted'] = temp_statesseqs[0]
                    Labels = {}
                    Accuracies = []
                    full_label_info = {}

                    for name, _ in labeled_data_train.groupby('state'):
                        print(name)
                        label = labeled_data_train.groupby('state').get_group(name)['predicted'].value_counts(normalize=True)
                        print(label)
                        Labels[name] = label.idxmax()
                        Accuracies.append(label.max())

                        for idx, acc in zip(label.index, label.values):
                            full_label_info[name+'_'+str(idx)] = [acc]
                    
                    
                    avg_acc = mean(Accuracies)
                    
                    print(Labels)
                    print(f'Accuracies: {Accuracies}')
                    print(f'Average Accuracy = {avg_acc}')

                    if SM.check_duplicate_label(Labels):

                        if avg_acc > max_accuracy:
                
                            max_accuracy = avg_acc
                            best_model, best_state_sequences, best_true_labels, best_states = temp_model, temp_statesseqs, temp_true_labels, temp_states                  
                            best_extra, best_kappa, best_iter = extra, kap, iter
                  
                            print(f'Model saved for {best_extra} extra states, kappa = {best_kappa}, and iters = {best_iter}')
                            self.save_model(best_model, device_name, model_name, filter, feature_names, save) 
                            self.save_label_accuracies(pd.DataFrame(full_label_info), device_name, model_name, filter, feature_names)
                            model_saved = True

        if model_saved == True:
            return best_model, best_state_sequences, best_true_labels, best_states
        else:
            return None, None, None, None

    def plot_states(self, colors):

        indexes_dict = labeled_data_test.groupby('state').indices
        
        for key in indexes_dict:
            new_list = []
           
            prev_ind = indexes_dict[key][0]
            new_list.append(prev_ind)

            for inds in indexes_dict[key][1:]:
                

                if inds - prev_ind > 1 : # plot backgroud color for new state
                    new_list.append(prev_ind)

                    plt.axvspan(new_list[0],new_list[1], facecolor=colors[key])

                    new_list = []
                    new_list.append(inds)

                if inds == indexes_dict[key][-1]: # plot backgroud color for las state

                    new_list.append(inds)

                    plt.axvspan(new_list[0],new_list[1], facecolor=colors[key])

                prev_ind = inds
    
    def plot_accuracy_likelihood(self, prediction_window, rolling_window):

        print(f"Steps {prediction_window} - MA Window {rolling_window}")
        scaler = MinMaxScaler() 
        accuracy_scaled = scaler.fit_transform(labeled_data_test[f'accuracy - {prediction_window} step'].rolling(rolling_window).mean().values.reshape(-1, 1))
        log_likelihood_scaled = scaler.fit_transform(labeled_data_test[f'log_likelihood - {prediction_window} step'].values.reshape(-1, 1))

        plt.figure(figsize=(12, 8), dpi=80)
        plt.plot(accuracy_scaled, color='black', label='Accuracy')
        plt.plot(log_likelihood_scaled, color='b', label='Log Likelihood')
        
        colors = {'game': 'salmon', 'augmented_reality': 'lightblue', 'idle': 'lightgreen', 'mining': 'peachpuff', 'stream': 'whitesmoke'}
        print(colors)
        self.plot_states(colors)

        plt.legend(loc='best')
        plt.grid()
        plt.show()

    def plot_confusion_matrix(self, confusion_matrix, axes, class_label, class_names, fontsize=14):

        df_cm = pd.DataFrame(confusion_matrix, index=class_names, columns=class_names)

        try:
            heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cbar=False, ax=axes)
        except ValueError:
            raise ValueError("Confusion matrix values must be integers.")
        heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
        heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
        axes.set_ylabel('True label')
        axes.set_xlabel('Predicted label')
        axes.set_title(class_label)

    def classification_report(self, confusion_matrix_list, labels):

        fig, ax = plt.subplots(1, 5, figsize=(12, 3))
        
        for axes, cfs_matrix, label in zip(ax.flatten(), np.average(confusion_matrix_list, axis=0), labels):
            
            self.plot_confusion_matrix(np.round(cfs_matrix).astype(int), axes, label, ["T", "F"])

            print(label)
            cfs_matrix = list(cfs_matrix)
            recall = cfs_matrix[0][0] / sum(cfs_matrix[0])
            spcificity = cfs_matrix[1][1] / sum(cfs_matrix[1])
            precision = cfs_matrix[0][0] / (cfs_matrix[0][0] + cfs_matrix[1][0])
            print('Recall', round(recall*100,2))
            print('Specificity', round(spcificity*100,2))
            print('Precision', round(precision*100,2))

            F1 = round(2 * (precision * recall) / (precision + recall),2)
            print('F1', F1)
        
        fig.tight_layout()
        plt.show()
    
#%%

# features = ['cpu_user_time_diff','cpu_system_time_diff','cpu_idle_time_diff','memory']#,'net_sent_diff']
# diff_columns = ['cpu_user_time', 'cpu_system_time','cpu_idle_time', 'net_sent', 'net_recv', 'io_counters_read_count_', 'io_counters_write_count_', 'io_counters_read_bytes_', 'io_counters_write_bytes_','io_counters_read_chars_', 'io_counters_write_chars_', 'cpu_times_user_','cpu_times_system_', 'cpu_times_children_user_', 'cpu_times_children_system_']

# device_name = 'RPi4B8GB' #RPi4B8GB, RPi4B4GB, RPi4B2GB2, RPi4B2GB1
# freq = 1800 # 1800, 1500, 1500, 1200
# feature_names = 'cpu-all_mem'
# progprint = 400
# #model_count = 4
# model_index = 3
# model_name = 'rvp_random_48hr'

#%%
#TODO: Save Training times for models manually
"""Designate model generation datasets""" 
# data set parameters
RPI_list = ["RPi4B8GB"] # "RPi4B8GB", "RPi4B4GB", "RPi4B2GB2", 
FREQ_list = ["1800"] # "1800", "1500", "1500", 
SEQ_list = ['random']#, 'random','pattern']
NUM_list = [''] #"_2",

step_sizes = [1,2,5,10,15,30,60]

# select the columns for the variables you want to use for multi-variate prediction
resos = ['cpu_user_time_diff', 'cpu_system_time_diff','cpu_idle_time_diff','memory']
resos_non_diff = ['cpu_user_time', 'cpu_system_time','cpu_idle_time','memory']

# Intialize the encoder-decoder model training parameters
filter_list = ['_sma_2'] #'_sg_5_2', '_sma_2', '_ema_1']
data_split = 0.7
progprint = [800]
kappa = [0.05, 0.1, 1.5]
extra_states = [2, 3, 4]
lookback = 300
model_index = 3
feature_names = 'cpu-all_mem'

"""Create instance of classes"""
SM = HDP_HSMM()
COMP = Comparator()

#%%
""" Run the models for all the datasets"""
skip = []

# load dataset
for RPI, FREQ in zip(RPI_list, FREQ_list):
    for NUM in NUM_list:
        for SEQ in SEQ_list:

            skip_count = 0
            hsmm_model = None
            model_name = f'rvp_{SEQ}_48hr{NUM}'
            
            try:

                labeled_data_train, labeled_data_test = SM.preprocess_data(RPI, FREQ, model_name, resos_non_diff, test = True)

            except:
                
                print(f"No dataset for {RPI}_{FREQ}MHz{SEQ}{NUM}")
                continue

            key = f"{RPI}_{FREQ}MHz_{SEQ}{NUM}"
            print(f"Training on {key}")

            for filter in filter_list:

                if filter == '': 
                    print("No Filter") 
                    resources = resos
                else: 
                    print("Filter:", filter)
                    # rename columns to include filter
                    resources = COMP.rename_columns(resos, filter)

                # create the model
                
                start_time = time.time()
                
                while hsmm_model == None:
                    skip_count += 1

                    if skip_count > 2:
                        print("Skipping Model")
                        skip.append(key)
                        break

                    else:    

                        model_path = 'HSMM_Models/{}/{}/{}_{}_{}.pickle'.format(RPI,filter,RPI,feature_names,model_name)

                        if os.path.exists(model_path):
                            
                            print("Loading Model")
                            hsmm_model, state_sequences, true_labels = SM.get_HSMM_state_seq(labeled_data_train, model_path, RPI)
                            training_time = 0
                        
                        else:
                            # print('skip model generation')
                            # hsmm_model == None
                            print("Generating Model")
                            hsmm_model, state_sequences, true_labels, states = SM.grid_search(labeled_data_train, RPI, model_name, filter, resources, feature_names, extra_states, kappa, progprint, "don't save fig") # number of extra states, kappa, number of sampling iterations  

                            training_time = time.time() - start_time
                    print("Training time: ",str(training_time))       

                if skip_count > 2:
                    continue

                
                #TODO: Comment below to for training

                #Place Modeled Hidden-States
                print('state sequences', state_sequences[0])
                print('training data', len(labeled_data_train))
                labeled_data_train['predicted'] = state_sequences[0]

                #Training Accuracy
                Labels = {}
                Accuracies = []

                for name,group in labeled_data_train.groupby('state'):
                    print(name)
                    label = labeled_data_train.groupby('state').get_group(name)['predicted'].value_counts(normalize=True)
                    print(label)
                    Labels[name] = label.idxmax()
                    Accuracies.append(label.max())

                #Create multi-step labels
                labeled_data_train['label'] = labeled_data_train['state'].map(Labels)
                labeled_data_test['label'] = labeled_data_test['state'].map(Labels)


                #create rolling window for prediction evaluations
                for step in step_sizes: 

                    # Check if model has already been used to predict
                    if os.path.exists(f'Results/HSMM/{RPI}/{filter}/predictions/{key}_{step}Step.csv'):
                        print(f'{key}_{step}Step.csv already exists. Skipping prediction.')
                        continue

                    labeled_data_test[f'label - {step} step'] = [list(map(int, window.to_list())) for window in labeled_data_test['label'].rolling(window=step)]
                    labeled_data_test[f'label - {step} step'] = labeled_data_test[f'label - {step} step'].shift(1-step)

                    #Prepares test data, predictions start after lookback period
                    prediction_start = lookback
                    test_labels = labeled_data_test['state'].values
                    df = labeled_data_test[resources]

                    observs_list, pred_observs_list = [], []
                    predicted_observations_list, observations_list = [], []
                    predicted_stateseq_list = []
                    test_labels_window_list = []
                    log_likelihoods_list = []
                    
                    observations_dict = {feature+'_observations':[] for feature in resources}
                    pred_observations_dict = {feature+'_predicted_observations':[] for feature in resources} 
                    

                    print(f"{step}-step prediction")
                    infer_times = []
                    for i in tqdm(range(prediction_start,len(labeled_data_test)+1)):
                        
                        seed_start_idx =  i - lookback
                        seed_end_idx = seed_start_idx + lookback
                        
                        if seed_end_idx > len(labeled_data_test)-step:
                            break
                        
                        # run predictions
                        infer_start_time = time.time()
                        predicted_observations, predicted_stateseq, log_likelihood = SM.HSMM_pred(hsmm_model,seed_start_idx, seed_end_idx, step)
                        infer_times.append(time.time() - infer_start_time)
                        
                        predicted_observations_list.append(predicted_observations[lookback:].tolist())
                        predicted_stateseq_list.append(predicted_stateseq[lookback:].tolist())
                        log_likelihoods_list.append(round(log_likelihood,2))
                        observations_list.append(df[i:i+step].values)

                        test_labels_window_list.append(test_labels[i:i+step])  

                    # store results
                    predicted_stateseq_list = ['lookback']*lookback + predicted_stateseq_list
                    log_likelihoods_list = ['lookback']*lookback + log_likelihoods_list


                    if len(labeled_data_test) != len(predicted_stateseq_list):
                        
                        if step != 1:
                            labeled_data_test = labeled_data_test[:-(step-1)]

                    labeled_data_test[f'predicted states - {step} step'] = predicted_stateseq_list
                    labeled_data_test[f'log_likelihood - {step} step'] = log_likelihoods_list

                    #For Observation Prediction 
                    pred_obs_dict, obs_dict = {}, {}

                    for resource in resources:
                        pred_obs_dict[resource], obs_dict[resource] = [], []

                    for feature in resources:
                        # Flatten the lists
                        pred_obs_flat = [val for sublist in predicted_observations_list[resources.index(feature)] for val in sublist]
                        obs_flat = [val for sublist in observations_list[resources.index(feature)] for val in sublist]

                        pred_obs_dict[feature].append(pred_obs_flat)
                        obs_dict[feature].append(obs_flat)

                    print(f"Lookback: {lookback}, Pred. Steps: {step}\n")

                    COMP.save_predictions(resources, "HSMM", key, pred_obs_dict, obs_dict, filter, step)

                    COMP.save_metrics(pred_obs_dict, obs_dict, training_time, infer_times, "HSMM", key, filter, step)

                    print("Done!")


print('Skipped datasets:', skip)            
#%%
# """Pre-process Data"""
# # data_name = model_name
# # prediction_windows = [1,2,5,10,15,30,60]
# # labeled_data_train, labeled_data_test = SM.preprocess_data(device_name, freq, data_name)
# #%%
# """Generates HSMM Model"""
# kappa_1 = 0.05
# model, state_sequences, true_labels, states  = SM.run_HSMM(labeled_data_train, features,extra_states = 1, kappa = kappa_1, progprint_xrange_var = progprint)
                 
# #%%
# """Generates HSMM Model using Grid Search"""
# model, state_sequences, true_labels, states = SM.grid_search([2], [0.1,0.1,0.1], [800], "don't save fig") # number of extra states, kappa, number of sampling iterations

# #%%
# """Reads previously generated model and extracts it"""
# model_path = 'HSMM_Models/{}/{}_{}_{}_{}kap_{}iter.pickle'.format(device_name.split('_')[0],device_name, feature_names, model_name,0.1,800)

# model, state_sequences, true_labels = SM.get_HSMM_state_seq(labeled_data_train, model_path, device_name)

# #%%
# """Place Modeled Hidden-States"""
# labeled_data_train['predicted'] = state_sequences[0]
# #%%
# """Training Accuracy"""
# Labels = {}
# Accuracies = []

# for name,group in labeled_data_train.groupby('state'):
#     print(name)
#     label = labeled_data_train.groupby('state').get_group(name)['predicted'].value_counts(normalize=True)
#     print(label)
#     Labels[name] = label.idxmax()
#     Accuracies.append(label.max())

# #%%
# """Create multi-step labels"""
# labeled_data_train['label'] = labeled_data_train['state'].map(Labels)
# labeled_data_test['label'] = labeled_data_test['state'].map(Labels)
# #create rolling window for prediction evaluations
# for pw in prediction_windows: 
#     labeled_data_test[f'label - {pw} step'] = [list(map(int,window.to_list())) for window in labeled_data_test['label'].rolling(window=pw)]
#     labeled_data_test[f'label - {pw} step'] = labeled_data_test[f'label - {pw} step'].shift(1-pw)


# #%%
# """Save pre-processed Dataset"""
# test_name = data_name
# labeled_data_train.to_csv(f"data/{device_name.split('_')[0]}/{device_name}_{freq}MHz_res_usage_data_train_pred_{test_name}.csv")

# #%%
# """Prepares test data, predictions start after lookback period"""
# prediction_start = lookback
# test_labels = labeled_data_test['state'].values
# df = labeled_data_test[features]


# #%%
# """Generates Label Predictions"""
# prediction_windows = [1,2,5,10,15,30,60]

# for prediction_window in prediction_windows:
#     observs_list, pred_observs_list = [], []
#     predicted_observations_list, observations_list = [], []
#     predicted_stateseq_list = []
#     test_labels_window_list = []
#     log_likelihoods_list = []
#     save_dict = {}
#     rmse_dict = {feature+'_rmse':[] for feature in features}
#     mae_dict = {feature+'_mae':[] for feature in features} 
#     observations_dict = {feature+'_observations':[] for feature in features}
#     pred_observations_dict = {feature+'_predicted_observations':[] for feature in features} 
#     rmse_dict_stat, mae_dict_stat = {}, {}

#     print(f"{prediction_window}-step prediction")

#     for i in tqdm(range(prediction_start,len(labeled_data_test)+1)):
        
#         seed_start_idx =  i - lookback
#         seed_end_idx = seed_start_idx + lookback
        
#         if seed_end_idx > len(labeled_data_test)-prediction_window:
#             break
        
#         predicted_observations, predicted_stateseq, log_likelihood = SM.HSMM_pred(df, model,seed_start_idx, seed_end_idx, prediction_window)
        
#         predicted_observations_list.append(predicted_observations[lookback:].tolist())
#         predicted_stateseq_list.append(predicted_stateseq[lookback:].tolist())
#         log_likelihoods_list.append(round(log_likelihood,2))
#         observations_list.append(df[i:i+prediction_window].values)

#         test_labels_window_list.append(test_labels[i:i+prediction_window])  

#     # store results
#     predicted_stateseq_list = ['lookback']*lookback + predicted_stateseq_list
#     log_likelihoods_list = ['lookback']*lookback + log_likelihoods_list


#     if len(labeled_data_test) != len(predicted_stateseq_list):
        
#         if prediction_window != 1:
#             labeled_data_test = labeled_data_test[:-(prediction_window-1)]

#     labeled_data_test[f'predicted states - {prediction_window} step'] = predicted_stateseq_list
#     labeled_data_test[f'log_likelihood - {prediction_window} step'] = log_likelihoods_list

#     """For Observation Prediction"""

#     idx = 0
#     for obs in observations_list:
    
#         obs_dict = SM.metrics_plots(obs, predicted_observations_list[idx], features) #predicted_stateseq, test_labels_window_list[idx], features)
#         idx += 1
#         for feature in features:
#             rmse_dict[feature+'_rmse'].append(obs_dict[feature+'_rmse'])
#             mae_dict[feature+'_mae'].append(obs_dict[feature+'_mae'])
#             observations_dict[feature+'_observations'].append(obs_dict[feature+'_observations'])
#             pred_observations_dict[feature+'_predicted_observations'].append(obs_dict[feature+'_predicted_observations'])

#     for key in rmse_dict:
#         #print('Avg',key,mean(rmse_dict[key]))
#         #print('Stdv',key,stdev(rmse_dict[key]))
#         rmse_dict_stat['Avg_'+key] = mean(rmse_dict[key])
#         rmse_dict_stat['Stdv_'+key] = stdev(rmse_dict[key])

#     rmse_dict_all = {**rmse_dict, **rmse_dict_stat}

#     for key in mae_dict:
#         #print('Avg',key,mean(mae_dict[key]))
#         #print('Stdv',key,stdev(mae_dict[key]))
#         mae_dict_stat['Avg_'+key] = mean(mae_dict[key])
#         mae_dict_stat['Stdv_'+key] = stdev(mae_dict[key])

#     mae_dict_all = {**mae_dict, **mae_dict_stat}

#     #%%
#     rmse_df = pd.DataFrame(rmse_dict_all)
#     mae_df = pd.DataFrame(mae_dict_all)
#     observations_df = pd.DataFrame(observations_dict)
#     pred_observations_df = pd.DataFrame(pred_observations_dict)

#     results = pd.concat([observations_df,pred_observations_df,rmse_df, mae_df], axis=1)
#     #%%
#     """Saves newly generated results"""
#     results.to_csv('Results/HSMM_Results_{}_{}sec_lookbk_{}sec_pred_window.csv'.format(device_name,lookback*5,prediction_window*5))

    
#     print(f"Lookback: {lookback}, Pred. Steps: {prediction_window}\n")
#     for feature in features:
#         observs_list, pred_observs_list = [], []
#         for index, row in results.iterrows():

#             if type(row[feature+'_observations']) == str:
#                 observs = loads(row[feature+'_observations'])
#                 pred_observs = loads(row[feature+'_predicted_observations'])
#             else:
#                 observs = row[feature+'_observations']
#                 pred_observs = row[feature+'_predicted_observations']
                
#             pred_observs = [0 if i < 0 else i for i in pred_observs]
#             observs_list.append(observs)
#             pred_observs_list.append(pred_observs)
        

#         print(feature,'mae' , round(mean_absolute_error(observs_list,pred_observs_list),3))
#         print(feature,'rmse', round(sqrt(mean_squared_error(observs_list,pred_observs_list)),3))
        
#         save_dict[feature+'_mae'] = [round(mean_absolute_error(observs_list,pred_observs_list),3)]
#         save_dict[feature+'_rmse'] = [round(sqrt(mean_squared_error(observs_list,pred_observs_list)),3)]
    
#     pd.DataFrame(save_dict).to_csv('Results/{}/HSMM_Error_Results_{}_{}sec_lookbk_{}sec_pred_window.csv'.format(device_name.split('_')[0],device_name,lookback*5,prediction_window*5))

#     read_flag = 0
#     print("Done!")


# #%%
# """Testing Accuracy"""
# for name,group in labeled_data_test.groupby('state'):
#     print(name)
#     print(labeled_data_test[lookback:].groupby('state').get_group(name)[f'predicted states - {1} step'].value_counts(normalize=True))
# #%%
# """Save test predictons"""
# test_name = data_name
# labeled_data_test.to_csv(f"data/{device_name.split('_')[0]}/{device_name}_{freq}MHz_res_usage_data_test_pred_{test_name}.csv")

# #%%
# """Read test predictions"""
# test_name = data_name
# labeled_data_test = pd.read_csv(f"data/{device_name.split('_')[0]}/{device_name}_{freq}MHz_res_usage_data_test_pred_{test_name}.csv")
# read_flag = 1

# #%%
# """remove lookback section"""
# labeled_data_test = labeled_data_test[lookback:]
# #%%
# """Analysis of Predictions"""

# plt.rc('font', **{'weight' : 'bold', 'size'   : 18})

# for prediction_window in prediction_windows:
#     accuracy, conf_matrix = [], []
#     for index, row in labeled_data_test.iterrows():
#         if read_flag == 0:
#             accuracy.append(accuracy_score(row[f'label - {prediction_window} step'], row[f'predicted states - {prediction_window} step'])*100)
#             conf_matrix.append(multilabel_confusion_matrix(row[f'label - {prediction_window} step'], row[f'predicted states - {prediction_window} step'], labels=labeled_data_test['label'].unique().tolist()))
#         else:
#             accuracy.append(accuracy_score(loads(row[f'label - {prediction_window} step']), loads(row[f'predicted states - {prediction_window} step']))*100)
#             conf_matrix.append(multilabel_confusion_matrix(loads(row[f'label - {prediction_window} step']), loads(row[f'predicted states - {prediction_window} step']), labels=labeled_data_test['label'].unique().tolist()))
        
#     labeled_data_test[f'accuracy - {prediction_window} step'] = accuracy
#     labeled_data_test[f'confusion matrix - {prediction_window} step'] = conf_matrix
   

#     print(f"{prediction_window} step prediction accuracy: {round(mean(accuracy),2)}%")
#     print(f"{prediction_window} step confusion matrix: ")

#     SM.classification_report(conf_matrix, labeled_data_test['state'].unique().tolist())

#     SM.plot_accuracy_likelihood(prediction_window, 100)

# #%%
# """Reads previously generated results"""
# results = pd.read_csv('Results/{}/HSMM_Results_{}_{}sec_lookbk_{}sec_pred_window.csv'.format(device_name.split('_')[0],device_name,lookback*5,prediction_window*5))

