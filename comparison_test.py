
#%%
import numpy as np
import pandas as pd
import time
from knn_tspi import KNeighborsTSPI
from encoder_decoder import Encoder_Decoder
from statsmodels.tsa.statespace.varmax import VARMAX
# from hdp_hsmm import HDP_HSMM
# from hbpshpo import HBPSHPO
import pickle
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import matplotlib.pyplot as plt
from tqdm import tqdm
from statistics import mean
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from scipy.signal import savgol_filter
import multiprocessing 

# In[57]:

class Comparator():

    def __init__(self):

        # Initialize the encoder-decoder model
        self.ED = Encoder_Decoder()

        # Initialize the scaler
        self.scaler = MinMaxScaler()
            

    def diff_data(self, df, columns, order=0.5):
        
        for col in columns:
            name1 = col + '_int_diff'
            name2 = col + f'_{order}order_diff'
            df[name2] = df[col]
            df[name1] = df[col].diff()
            for i in range(1, df.shape[0]):
                df.loc[i, name2] = df.loc[i-1, name2] + order*(df.loc[i, col]-df.loc[i-1, col]-df.loc[i-1, name2])
        return df

    def filter_data(self, df, columns, savgol_window_size = 5, savgol_degree = 2, sma_window_size = 2, ema_window_size = 1):
        
        for resource in columns:
            ts = df[resource].reset_index(drop=True)
            
            #Savistky-Golay Filter
            ts_sg = pd.DataFrame(savgol_filter(ts, savgol_window_size, savgol_degree))
            ts_sg[ts_sg < 0] = 0 
            df[resource+f'_sg_{savgol_window_size}_{savgol_degree}'] = ts_sg

            #Simple Moving Average Filter
            ts_sma = ts.rolling(sma_window_size, min_periods=1).mean()
            df[resource+f'_sma_{sma_window_size}'] = ts_sma

            #Exponential Moving Average Filter
            ts_ema = ts.ewm(ema_window_size).mean()
            df[resource+f'_ema_{ema_window_size}'] = ts_ema
        
        return df

    def calculate_smape(self, actual, predicted):
    
        # Convert actual and predicted to numpy
        # array data type if not already
        if not all([isinstance(actual, np.ndarray), 
                    isinstance(predicted, np.ndarray)]):
            actual, predicted = np.array(actual),
            np.array(predicted)
    
        return round(
            np.mean(
                np.abs(predicted - actual) / 
                ((np.abs(predicted) + np.abs(actual))/2)
            )*100, 2
        )

    def save_metrics(self, predictions, obs, training_time, inference_times, model_type, data, filter, step):

        metrics = {}
        tmp_inference_times = inference_times

        for resource, preds in predictions.items():
            
            preds= np.array(preds)
            obs_res = np.array(obs[resource])

            #check if tmp_inference_times is a dict:
            if isinstance(tmp_inference_times, dict):
                infer_times = tmp_inference_times[resource]
            else:
                infer_times = tmp_inference_times

            # print resource specific metrics
            print(f"{resource} Metrics:")
            
            # Overall metrics        
            # print R2
            score = round(r2_score(preds.ravel(), obs_res.ravel()),3)
            print(f"{resource} Score:", score)
            
            #print time-step specific RMSE
            rmse = round(mean_squared_error(preds.ravel(), obs_res.ravel(), squared=False),3)
            print(f"{resource} RMSE for ", rmse)

            #print time-step specific MAE
            mae = round(mean_absolute_error(preds.ravel(), obs_res.ravel()),3)
            print(f"{resource} MAE for ",mae)

            #print MAPE
            mape = round(mean_absolute_percentage_error(preds.ravel(), obs_res.ravel())*100,2)
            print(f"{resource} MAPE for ", mape, "%")

            #print SMAPE
            smape= self.calculate_smape(preds.ravel(), obs_res.ravel()) 
            print(f"{resource} SMAPE for ", smape, "%")

            # Store metrics in a dictionary
            metrics[resource] = {
                "Resource": resource,
                "Score": score,
                "RMSE": rmse,
                "MAE": mae,
                "MAPE": mape,
                "SMAPE": smape,
                "Training time": training_time,
                "Mean Inference time": mean(infer_times),
                "Max Inference time": max(infer_times),
                "Min Inference time": min(infer_times)
            }

        rpi = data.split('_')[0]

        os.makedirs(os.path.join('Results', model_type, rpi, filter,'metrics'), exist_ok=True)


        # Convert metrics to a pandas DataFrame
        df_metrics = pd.DataFrame.from_dict(metrics, orient='index')

        print('Saving metrics to {}_step.csv'.format(step))
        # Save metrics to a CSV file
        df_metrics.to_csv(r'Results/{}/{}/{}/metrics/{}_{}Step.csv'.format(model_type,rpi,filter, data,step), index=False)




    def plot_learning_curve(self, loss, loss_type, step):
        # Plot the learning curve
        plt.figure(figsize=(10, 10))
        plt.plot(loss, label=loss_type)
        plt.title(f"Learning Curve for {step} Step Prediction")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()
        

    def min_max_norm(self, data, scaler):
        # Fit and transform the training data
        n1_samples, n1_timesteps, n1_features = data.shape[0], data.shape[1], data.shape[2]
        data = scaler.fit_transform(data.reshape(n1_samples * n1_timesteps, n1_features))
        return data.reshape(n1_samples, n1_timesteps, n1_features)
        

    def rescale_norm(self, data, scaler):
        # Transform the test data using the fitted scaler
        n2_samples, n2_timesteps, n2_features = data.shape[0], data.shape[1], data.shape[2]
        data = scaler.inverse_transform(data.reshape(n2_samples * n2_timesteps, n2_features))
        return data.reshape(n2_samples, n2_timesteps, n2_features)

    def rename_columns(self, resos, filt):
        new_names = []
        for reso in resos:
            new_names.append(reso+filt)
        return new_names


    def save_model(self, model, model_type, data, step, filter, resource):

        step = str(step)
        rpi = data.split('_')[0]

        if filter == '':
            filter = 'NoFilter'
        
        # save varma model
        if model_type == 'VARMA':
        
            os.makedirs(os.path.join(r'{}_Models'.format(model_type), rpi, filter), exist_ok=True)

            print('Saving VARMA model {}_cpu-all-mem.pickle'.format(data))
            with open(r'{}_Models/{}/{}/{}_cpu-all-mem.pickle'.format(model_type,rpi,filter,data),'wb') as outfile:
                pickle.dump(model,outfile,protocol=-1)
            
        elif model_type == 'Encoder-Decoder':

            os.makedirs(os.path.join(r'{}_Models'.format(model_type), rpi, filter,r'Step_{}'.format(step)), exist_ok=True)
            
            print('Saving ED model {}_cpu-all-mem.pickle'.format(data))
            with open(r'{}_Models/{}/{}/Step_{}/{}_cpu-all-mem.pickle'.format(model_type,rpi,filter,step,data),'wb') as outfile:
                pickle.dump(model,outfile,protocol=-1)
        
        elif model_type == 'kNN-TSPI':
            
            os.makedirs(os.path.join(r'{}_Models'.format(model_type), rpi, filter,r'Step_{}'.format(step)), exist_ok=True)
            
            print('Saving kNN-TSPI model {}_{}.pickle'.format(data,resource))
            with open(r'{}_Models/{}/{}/Step_{}/{}_{}.pickle'.format(model_type,rpi,filter,step,data,resource),'wb') as outfile:
                pickle.dump(model,outfile,protocol=-1)

        elif model_type == 'HDP-HSMM':

            os.makedirs(os.path.join(r'{}_Models'.format(model_type), rpi, filter), exist_ok=True)
            print('Saving HDP-HSMM model {}_cpu-all-mem.pickle'.format(data))
            with open(r'{}_Models/{}/{}/{}_cpu-all-mem.pickle'.format(model_type,rpi,filter,data),'wb') as outfile:
                pickle.dump(model,outfile,protocol=-1)

        outfile.close() 

    def save_predictions(self, resources, model_type, data,predictions, Y_tests, filter, step):
        
        if filter =='':
            filter = 'NoFilter'

        rpi = data.split('_')[0]
        results_df = pd.DataFrame()
        for resource in resources:
            results_df[f"{resource}_preds{filter}"] = predictions[resource]
            results_df[f"{resource}_obs{filter}"] = Y_tests[resource]
        
        os.makedirs(os.path.join('Results', model_type, rpi, filter, 'predictions'), exist_ok=True)
        print('Saving predictions to {}_{}Step.csv'.format(data, step))
        results_df.to_csv(r'Results/{}/{}/{}/predictions/{}_{}Step.csv'.format(model_type,rpi,filter, data, step), index=False)

    def read_dataset(self, RPI, FREQ, NUM, SEQ, drop_list):
        
        
        file_path = f"~/git_repos/u-worc/data/{RPI}/{RPI}_{FREQ}MHz_res_usage_data_train_pred_rvp_{SEQ}_48hr{NUM}.csv"
        df=pd.read_csv(file_path)
        df.drop(drop_list, inplace=True, axis=1)
        df.dropna()
        
        return df

    def reorganize_datapoints(self, time_specific_datapoints, datapoints_dict, resources, step_size):
        
        # reorganize predictions to be in the same format as Y_tests
        for col_indx, resource in enumerate(resources):
            split_time_specific_datapoints = []

            for row_indx in range(step_size):
                split_time_specific_datapoints.append(time_specific_datapoints[row_indx][col_indx].astype(list))

            datapoints_dict[resource].append(split_time_specific_datapoints)

        return datapoints_dict

    def VARMA_predict(self, key, filter, split, df, resos, step_sizes):

        import warnings
        from statsmodels.tools.sm_exceptions import ConvergenceWarning
        warnings.filterwarnings("ignore", category=ConvergenceWarning)


        model_type = 'VARMA'
        print("Model Type: ", model_type)


        if filter == '': 
            print("No Filter") 
            resources = resos
        else: 
            print("Filter:", filter)
            # rename columns to include filter
            resources = self.rename_columns(resos, filter)

        # split dataset into train, test, eval
        train = df[resources][:int(split*len(df))]
        test = df[resources][int(split*len(df)):]

        #Train VARMA model
        start_time = time.time()
        var_model = VARMAX(endog=train)
        var_model_fit = var_model.fit(disp=0)
        
        training_time = time.time() - start_time
        print("Training time: ",str(training_time))
        
        self.save_model(var_model, model_type, key, 0, filter, 0)


        for step in step_sizes:

            # Check if model has already been trained
            rpi = key.split('_')[0]
            if os.path.exists(f'Results/{model_type}/{rpi}/{filter}/predictions/{key}_{step}Step.csv'):
                print(f'{key}_{step}Step.csv already exists. Skipping training and prediction.')
                continue
            
            print("Step size: ", step)
            X_train, Y_train, X_test, Y_test = self.ED.split_test_train_seq(train, test, 300, step, len(resources))
            
            # initialize predictions and Y_test dictionaries
            predictions, Y_tests = {}, {}
            for resource in resources:
                predictions[resource], Y_tests[resource] = [], []

            infer_times, retrain_times = [], []
            #474 m 49.9 s
            for t in tqdm(range(0,len(X_test))):

                # make prediction
                infer_start_time = time.time()
                yhat = var_model_fit.forecast(steps=step,disp=0)
                infer_times.append(time.time() - infer_start_time)

                # append prediction by resource to predictions dictionary
                yhat_split = np.split(yhat, len(resources), axis=1)
                for i, resource in enumerate(resources):
                    predictions[resource].append(yhat_split[i])

                # retrain the model
                retrain_start_time = time.time()
                var_model = VARMAX(endog=X_test[t])
                var_model_fit = var_model.fit(disp=0)
                retrain_times.append(time.time() - retrain_start_time)

            # Calculate Inference Time
            print("Inference time: ",(mean(infer_times)))
            print("Retrain time: ",(mean(retrain_times)))

            # Split Y_test and append to Y_tests by individual resources
            for Y in Y_test:
                Y_test_split = np.split(Y, len(resources), axis=1)
                for i, resource in enumerate(resources):
                    Y_tests[resource].append(Y_test_split[i])

            # Save predictions and Y_tests to df
            self.save_predictions(resources, model_type, key, predictions, Y_tests, filter, step)
            
            # Save and Print Metrics
            self.save_metrics(predictions, Y_tests, training_time, infer_times, model_type, key, filter, step)


    def _kNN_TSPI_parallel(self, resource, train, test, step, key, filter, model_type):


        print("Training model for: ", resource)
        knn_model = KNeighborsTSPI()
        start_training = time.time()
        knn_model.fit(data=train[resource].values)
        training_time = time.time() - start_training
        print("Training Time:", training_time)

        self.save_model(knn_model, model_type, key, step, filter, resource)

        Y_test, preds, infer_times = [], [], []
        _, _, X_test, Y_test = self.ED.split_test_train_seq(train[resource], test[resource], 300, step, 1)

        print(resource)

        for i in tqdm(range(0, len(X_test))):
            y_hats = []
            for j in range(step):
                pred_time = time.time() 
                yhat = knn_model.predict(h=1)  
                y_hats.append(yhat)
                infer_times.append(time.time() - pred_time)
                knn_model.fit(data=np.append(np.roll(X_test[i],-1)[:-1],yhat))
            preds.append(y_hats)

        print(f"{model_type}  Average Inference Time for ", resource, mean(infer_times))            
        Y_tests = {'cpu_user_time_diff':[], 'cpu_system_time_diff':[],'cpu_idle_time_diff':[],'memory':[],
                   'cpu_user_time_diff_sg_5_2':[], 'cpu_system_time_diff_sg_5_2':[],'cpu_idle_time_diff_sg_5_2':[],'memory_sg_5_2':[],
                   'cpu_user_time_diff_ema_1':[], 'cpu_system_time_diff_ema_1':[],'cpu_idle_time_diff_ema_1':[],'memory_ema_1':[],
                   'cpu_user_time_diff_sma_2':[], 'cpu_system_time_diff_sma_2':[],'cpu_idle_time_diff_sma_2':[],'memory_sma_2':[]
                   }
        # Split Y_test and append to Y_tests by individual resources
        for Y in Y_test:
            Y_tests = self.reorganize_datapoints(Y, Y_tests, [resource], step)

        return Y_tests[resource], preds, training_time, infer_times
    
    def _parallel_wrapper(self, args):
        resource, train, test, step, key, filter, model_type = args
        return self._kNN_TSPI_parallel(resource, train, test, step, key, filter, model_type)



    def kNN_TSPI_predict(self, key, filter, split, df, resos, step_sizes, parallel = True):
        
        model_type = 'kNN-TSPI'
        
        print("Model Type: ", model_type)

        if filter == '': 
            print("No Filter") 
            resources = resos
        else: 
            print("Filter:", filter)
            # rename columns to include filter
            resources = self.rename_columns(resos, filter)

        # split dataset into train, test, eval
        train = df[resources][:int(split*len(df))]
        test = df[resources][int(split*len(df)):]

        for step in step_sizes:

            # Check if model has already been trained
            rpi = key.split('_')[0]
            if os.path.exists(f'Results/{model_type}/{rpi}/{filter}/predictions/{key}_{step}Step.csv'):
                print(f'{key}_{step}Step.csv already exists. Skipping training and prediction.')
                continue

            
            Y_tests, preds, infer_times = {}, {}, {}
            for resource in resources:
                Y_tests[resource] = []
                preds[resource] = []
                infer_times[resource] = []
            
            print("Step size: ", step)

            # In parallel
            if parallel == True:

                args_list = [(resource, train, test, step, key, filter, model_type) for resource in resources]

                with multiprocessing.get_context('spawn').Pool(processes=len(resources)) as pool:
                    results = pool.map(self._parallel_wrapper, args_list)

                for res, (yt, p, tt, it) in zip(resources, results):
                    Y_tests[res] = yt
                    preds[res] = p
                    training_time = tt
                    infer_times[res] = it
            
            # In series
            else:
                for resource in resources:
                    
                    print("Training model for: ", resource)
                    knn_model = KNeighborsTSPI()
                    start_training = time.time()
                    knn_model.fit(data=train[resource].values)
                    training_time = time.time() - start_training
                    print("Training Time:", training_time)

                    self.save_model(knn_model, model_type, key, step, filter, resource)

                    Y_tests[resource], preds[resource], infer_times[resource] = [], [], []
                    _, _, X_test, Y_test = self.ED.split_test_train_seq(train[resource], test[resource], 300, step, 1)
                            
                    print(resource)

                    for i in tqdm(range(0, len(X_test))):
                        y_hats = []
                        for j in range(step):
                            pred_time = time.time() 
                            yhat = knn_model.predict(h=1)  
                            y_hats.append(yhat)
                            infer_times[resource].append(time.time() - pred_time)
                            knn_model.fit(data=np.append(np.roll(X_test[i],-1)[:-1],yhat))
                        preds[resource].append(y_hats)
            
                    print(f"{model_type}  Average Inference Time for ", resource, mean(infer_times[resource]))            
                    
                    # Split Y_test and append to Y_tests by individual resources
                    for Y in Y_test:
                        Y_tests = self.reorganize_datapoints(Y, Y_tests, [resource], step)    

            # Save predictions and Y_tests to df
            self.save_predictions(resources, model_type, key, preds, Y_tests, filter, step)
            
            # Save and Print Metrics
            self.save_metrics(preds, Y_tests, training_time, infer_times, model_type, key, filter, step)


    def ED_predict(self, key, filter, split, df, resos, step_sizes, lr_list, neuron_mult_list, num_epoch):

        model_type = 'Encoder-Decoder'
        
        print("Model Type: ", model_type)

        if filter == '': 
            print("No Filter") 
            resources = resos
        else: 
            print("Filter:", filter)
            # rename columns to include filter
            resources = self.rename_columns(resos, filter)

        # split dataset into train, test, eval
        train = df[resources][:int(split*len(df))]
        test = df[resources][int(split*len(df)):]

        

        for step, lr_itr, neuron_mult, num_epoch_itr in zip(step_sizes, lr_list, neuron_mult_list, num_epoch):

            print("Step size: ", step)
            
            # Check if model has already been trained
            rpi = key.split('_')[0]
            if os.path.exists(f'Results/{model_type}/{rpi}/{filter}/predictions/{key}_{step}Step.csv'):
                print(f'{key}_{step}Step.csv already exists. Skipping training and prediction.')
                continue

            
            in_steps = step*2
            X_train, Y_train, X_test, Y_test  = self.ED.split_test_train_seq(train, test, in_steps, step, len(resources))
            
            predictions, Y_tests = {}, {}

            for resource in resources:
                predictions[resource], Y_tests[resource] = [], []
            
            # Fit and transform the training data
            X_train = self.min_max_norm(X_train,self.scaler)
            Y_train = self.min_max_norm(Y_train,self.scaler)
            
            # Transform the test data using the fitted scaler
            X_test = self.min_max_norm(X_test,self.scaler)
            print("Training ED model for learning-rate:", lr_itr)
            ed_model = self.ED.bid_lstm_model(in_steps, step, len(resources), lr_itr, neuron_mult)

            start_training = time.time()
            # fit model
            ed_model.fit(X_train, Y_train, epochs=num_epoch_itr, verbose=1)#, validation_data=(X_test, Y_test))
            training_time = time.time() - start_training
            print(step, "Step Training Time:", training_time)

            self.save_model(ed_model, model_type, key, step, filter, resource)

            # print learning curve
            self.plot_learning_curve(ed_model.history.history['loss'],'loss',step)
            # plot_learning_curve(ed_model.history.history['val_loss'],'val_loss',step)

        
            infer_times = []

            for i in tqdm(range(len(X_test))):

                tmp = np.array(X_test[i])
                tmp = np.expand_dims(tmp, axis=0)

                pred_time = time.time()
                
                yhat = ed_model.predict(tmp, verbose=0)
                
                infer_times.append(time.time()-pred_time)

                # append rescaled prediction by resource to predictions dictionary
                yhat = self.rescale_norm(yhat, self.scaler)

                predictions = self.reorganize_datapoints(yhat[0], predictions, resources, step)
            
                

            print(f"{model_type} Average Inference Time", mean(infer_times))

            # Split Y_test and append to Y_tests by individual resources
            for Y in Y_test:
                Y_tests = self.reorganize_datapoints(Y, Y_tests, resources, step)   

            # Save predictions and Y_tests to df
            self.save_predictions(resources, model_type, key, predictions, Y_tests, filter, step)
            
            # Save and Print Metrics
            self.save_metrics(predictions, Y_tests, training_time, infer_times, model_type, key, filter, step)          
                



#%%

    # #load test dataset
    # df=pd.read_csv(r'~/git_repos/u-worc/data/RPi4B8GB/RPi4B8GB_1800MHz_res_usage_data_train_pred_rvp_pattern_48hr.csv')
    # rpi = 'RPi4B8GB_1800'
    # seq_type ='pattern'
    # df.drop(['time_stamp','net','gpu','gpu_mem','link_quality_max','wifi_freq','state','predicted', 'label'], inplace=True, axis=1)
    # df.shape

#%%
    #For HDP-HSMM
    resos_non_diff = ['cpu_user_time', 'cpu_system_time','cpu_idle_time','memory']

    # Intialize the encoder-decoder model training parameters
    filter_list = ['','_sg_5_2', '_sma_2', '_ema_1']
    data_split = 0.7
    progprint = 800
    lookback = 300
    model_index = 3
    feature_names = 'cpu-all_mem'
    skipped = []
    """Create instance of classes"""
    # SM = SemiMarkov()

#%%    
if __name__ == '__main__':

    # data set parameters
    RPI_list = ["RPi4B8GB", "RPi4B4GB", "RPi4B2GB2", "RPi4B2GB1"]
    FREQ_list = ["1800", "1500","1500", "1200"]
    SEQ_list = ['random', 'pattern'] #'random', 'pattern'
    NUM_list =["", "_2","_3", "_4"] # "", "_2",

    drop_list = ['net','gpu','gpu_mem','link_quality_max','wifi_freq','state','predicted', 'label']

    # select the columns for the variables you want to use for multi-variate prediction
    resos = ['cpu_user_time_diff', 'cpu_system_time_diff','cpu_idle_time_diff','memory']


    # Intialize the encoder-decoder model training parameters
    filter_list = ['','_sg_5_2', '_sma_2', '_ema_1']
    step_sizes = [15]#1,2,5,10,15,30,60] # 

    # ED model parameters
    neuron_mult_list = [2]#1,1,1,2,2,3,4] #
    lr_list = [0.0001]#0.01, 0.01, 0.01, 0.001, 0.001, 0.00001, 0.000001] # 
    num_epoch = [15]#10, 10, 10, 15, 15, 15, 15] #

    split = 0.7
    COMP = Comparator()

    # initialize the dataframe to store resouce usage
    df = pd.DataFrame()

    """ Run the models for all the datasets"""
    # load dataset
    for RPI, FREQ in zip(RPI_list, FREQ_list):
        for NUM in NUM_list:
            for SEQ in SEQ_list:
                
                try:

                    df =  COMP.read_dataset(RPI, FREQ, NUM, SEQ, drop_list)
                except:
                    
                    print(f"No dataset for {RPI}_{FREQ}MHz{SEQ}{NUM}")
                    continue

                key = f"{RPI}_{FREQ}MHz_{SEQ}{NUM}"

                print("Dataset: ", key)

                # filter the dataset
                df = COMP.filter_data(df, resos)

                for filter in filter_list:

                    # #split df into chuncks of size 5000
                    # chunk_size = 5000
                    # df_chunks = [df[i:i+chunk_size] for i in range(0, len(df), chunk_size)]
                    # for i, df in enumerate(df_chunks):
                    #     key = key + f'_chunk_{i}'
                    #     COMP.VARMA_predict(key, filter, split, df, resos, step_sizes)


                    # COMP.kNN_TSPI_predict(key, filter, split, df, resos, step_sizes)
                    
                    COMP.ED_predict(key, filter, split, df, resos, step_sizes, lr_list, neuron_mult_list, num_epoch)

            


# %%

# #%%


    
# # Plot results in bar chart, all metrics per step size
# fig, ax = plt.subplots(2,2, figsize=(15, 15))
# fig.suptitle('Error Metrics for Each Model', fontsize=16)
# #VARMA
# for num, filter in enumerate(filter_list):
#     num1, num2 = get_nums(num)
#     ax[num1][num2].set_title(f'VARMA - {filter}')
#     ax[num1][num2].set_ylabel('Error')
#     ax[num1][num2].set_xlabel('Step Size')
#     ax[num1][num2].bar(vr_err_dict[filter]['Step size'], vr_err_dict[filter]['RMSE'], label='RMSE')
#     ax[num1][num2].bar(vr_err_dict[filter]['Step size'], vr_err_dict[filter]['MAE'], label='MAE')
#     ax[num1][num2].errorbar(vr_err_dict[filter]['Step size'], vr_err_dict[filter]['MAPE'], label='MAPE')
#     ax[num1][num2].errorbar(vr_err_dict[filter]['Step size'], vr_err_dict[filter]['Score'], label='R2')
#     ax[num1][num2].set_xticks(vr_err_dict[filter]['Step size'])
#     ax[num1][num2].legend(loc="upper left")

# #%%
# #kNN-TSPI
# ax[2].set_title('kNN-TSPI')
# ax[2].set_ylabel('Error')
# ax[2].set_xlabel('Step Size')
# ax[2].bar(kn_err_dict[filter]['Step size'], kn_err_dict[filter]['RMSE'], label='RMSE')
# ax[2].bar(kn_err_dict[filter]['Step size'], kn_err_dict[filter]['MAE'], label='MAE')
# ax[2].errorbar(kn_err_dict[filter]['Step size'], kn_err_dict[filter]['MAPE'], label='MAPE')
# ax[2].errorbar(kn_err_dict[filter]['Step size'], kn_err_dict[filter]['Score'], label='R2')
# ax[2].set_xticks(kn_err_dict[filter]['Step size'])
# ax[2].legend()

# plt.show()


# import os
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from math import pi, log10

# def plot_radar_chart(csv_dir):
    # """
    #     This function takes a list of csv file paths as input, reads the contents of each csv file and generates a radar chart 
    #     for each csv file. The radar chart displays the normalized values of several performance metrics including RMSE, MAE, 
    #     MAPE, SMAPE, Average Inference Time, Maximum Inference Time, and Logarithm of Training Time. The radar chart shows the 
    #     performance of each model for each metric. The area of the radar chart can be used to compare the overall performance of 
    #     each model across all metrics.

    #     :param csv_files: A list of paths to csv files containing performance metrics for each model.
    #     :return: A list of areas of the radar charts generated for each csv file.
    #     """
#     df_list = []
#       
#       #Loop through each file in the specified directory that ends with .csv extension
#     for filename in os.listdir(csv_dir):
#         if filename.endswith('.csv'):
#             path = os.path.join(csv_dir, filename)
#             df = pd.read_csv(path)
#             df = df.set_index('Resource')
#             df_list.append(df)

         # Define the metrics to be plotted
#     metrics = ['Score', 'RMSE', 'MAE', 'MAPE', 'SMAPE', 'Training time', 'Mean Inference time', 'Max Inference time']
#     n_metrics = len(metrics)
#     n_models = len(df_list)
#       #Calculate the angle for each metric
#     angles = [n / float(n_metrics) * 2 * pi for n in range(n_metrics)]
#     angles += angles[:1]
#     max_value = 1.1
#       # Create a new figure with a polar subplot for each model
#     fig, axs = plt.subplots(n_models, 1, figsize=(8, 6 * n_models), subplot_kw=dict(polar=True))

#     area_list = []
#     for i, df in enumerate(df_list):
#         # normalize the values
#         norm_df = (df - df.min()) / (df.max() - df.min())
#         norm_df['Training time'] = norm_df['Training time'].apply(lambda x: log10(x))

#         # add the average and maximum inference time columns
#         norm_df['Avg Inference time'] = norm_df[['Mean Inference time', 'Max Inference time']].mean(axis=1)
#         norm_df['Max Inference time'] = norm_df['Max Inference time']
#         norm_df = norm_df.drop('Mean Inference time', axis=1)

#           #Get the values for each metric from the data frame            
#         values = norm_df.loc[metrics].values.flatten().tolist()
#         values += values[:1]
#           # Calculate the area of the radar chart for the model
#         area = np.pi * np.power(max_value, 2) / n_metrics * np.sum([values[i] * np.power(max_value, 2) / n_metrics * (angles[i+1] - angles[i]) for i in range(n_metrics)])
#         area_list.append(area)

#           #Plot the radar chart for the model
#         ax = axs[i]
#         ax.plot(angles, values, linewidth=1, linestyle='solid')
#         ax.fill(angles, values, alpha=0.1)
#         ax.set_title(os.path.splitext(filename)[0])
#         ax.set_xticks(angles[:-1])
#         ax.set_xticklabels(metrics)
#         ax.set_yticks(np.linspace(0, 1, 6))
#         ax.set_yticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1'])
#         ax.yaxis.grid(True)

#     plt.show()

#     return area_list



# # %%

# import scipy.stats as stats

# # create a dataframe with performance metrics for different models
# metrics_df = 0

# # transpose dataframe to have models as rows and metrics as columns
# metrics_df = metrics_df.transpose()
# metrics_df.columns = ['Metric 1', 'Metric 2', 'Metric 3', 'Metric 4']

# # calculate Friedman test
# friedman_stat, p_value = stats.friedmanchisquare(*metrics_df.values)

# # display Friedman test results
# print(f"Friedman statistic = {friedman_stat:.3f}")
# print(f"P-value = {p_value:.3f}")

# # calculate critical difference for Nemenyi post-hoc test
# num_models = metrics_df.shape[0]
# num_metrics = metrics_df.shape[1]
# k = num_metrics  # number of repetitions
# q = stats.distributions.t.ppf(1-0.05/2, num_models-1) * np.sqrt(k*(k+1)/(6*num_models))
# cd = q * np.sqrt(num_models*(num_models+1)/(6*num_metrics))

# # perform Nemenyi post-hoc test
# ranks = stats.rankdata(-metrics_df.values, axis=0)
# average_ranks = np.mean(ranks, axis=1)
# t_values = np.abs(average_ranks[:, np.newaxis] - average_ranks)
# nemenyi_p_values = stats.distributions.norm.sf(t_values / cd)

# # display Nemenyi post-hoc test results
# print("Nemenyi post-hoc test results:")
# print("-"*30)
# for i in range(num_models):
#     for j in range(i+1, num_models):
#         print(f"Comparison between Model {i+1} and Model {j+1}:")
#         print(f"T-value = {t_values[i,j]:.3f}")
#         print(f"P-value = {nemenyi_p_values[i,j]:.3f}")
#         if nemenyi_p_values[i,j] < 0.05:
#             print("Reject null hypothesis: models are significantly different")
#         else:
#             print("Accept null hypothesis: models are not significantly different")
#         print("-"*20)


#%%
#Back up

# In[59]:

# print("HDP-HSMM Testing")

# RUMP = HDP_HSMM()
# look_back = 300
# progprint = 200
# kappa_1 = 0.05

# # select the columns for the variables you want to use for multi-variate prediction
# resos = ['cpu_user_time_diff','cpu_idle_time_diff','memory']
# filter_list = ['','_sg_5_2', '_sma_2', '_ema_1']
# step_sizes = [1,2,5,10,15,30,60]

# df=pd.read_csv(r'~/Desktop/u-worc/data/RPi4B8GB/RPi4B8GB_1800MHz_res_usage_data_train_pred_rvp_pattern_48hr.csv')
# df.drop(['net','gpu','gpu_mem','link_quality_max','wifi_freq', 'predicted', 'label'], inplace=True, axis=1)
# df = filter_data(df, resos)

# for filter in filter_list:

#     if filter == '': 
#         print("No Filter") 
#         resources = resos
#     else: 
#         print("Filter:", filter)
#         # rename columns to include filter
#         resources = rename_columns(resos, filter)


#     # split dataset into train, test, eval
#     train = df[resources+['state']][:int(0.8*5000)]
#     test = df[resources+['state']][int(0.8*5000):5000]
#     eval = df[resources+['state']][5000:6000]

#     # train the model
#     start_training = time.time()
#     hdp_hsmm_model, state_sequences, true_labels, states  = RUMP.run_HSMM(train, extra_states = 1, kappa = kappa_1, progprint_xrange_var = progprint)
#     print("Training Time:", time.time() - start_training)

#     for step in step_sizes:

#         preds, infer_times = {}, {}
        
#         print("Step size: ", step)
               

#         X_train, Y_train, X_test, Y_test, X_eval, Y_eval = ED.split_test_train_seq(train, test, eval, look_back, step, len(resources))

#         # initialize predictions and Y_test dictionaries
#         predictions, Y_tests = {}, {}
#         for resource in resources:
#             predictions[resource], Y_tests[resource] = [], []
        
#         for i in tqdm(range(0,len(test)+1)):
        
            
#             pred_time = time.time() 
#             yhat, _, _ = RUMP.HSMM_pred(test[i:i+look_back], hdp_hsmm_model, step)
#             infer_times.append(time.time() - pred_time)

#             # append prediction by resource to predictions dictionary
#             yhat_split = np.split(yhat[look_back:], len(resources), axis=1)
#             for i, resource in enumerate(resources):
#                 predictions[resource].append(yhat_split[i])
            
    
#         print("HDP-HSMM Average Inference Time for ", resource, mean(infer_times[resource]))
        

#         # Split Y_test and append to Y_tests by individual resources
#         for Y in Y_test:
#             Y_test_split = np.split(Y, len(resources), axis=1)
#             for i, resource in enumerate(resources):
#                 Y_tests[resource].append(Y_test_split[i])

#         # Calculate Metrics
#         for resource, preds in predictions.items():
#             print_metrics(np.array(preds), np.array(Y_tests[resource]), resource)

# # In[59]:
# import warnings
# warnings.filterwarnings('ignore', message='The objective has been evaluated at this point before.')

# print("HBPSHPO Testing")
# DL_NAS = HBPSHPO()
# #scaler = MaxAbsScaler()

# dataset = np.nan_to_num(df.values).astype('float32')
# scaler.fit(dataset)
# DL_NAS.dataset = scaler.transform(dataset)

# for filter in filter_list:

#     if filter == '': 
#         print("No Filter") 
#         resources = resos
#     else: 
#         print("Filter:", filter)
#         # rename columns to include filter
#         resources = rename_columns(resos, filter)

#     models = {}

#     # split dataset into train, test, eval
#     train = df[resources][:int(0.8*5000)]
#     test = df[resources][int(0.8*5000):5000]
#     eval = df[resources][5000:6000]

#     for step in step_sizes:

#         #lower-bounds and upper-bounds of hyperparameters
#         lb = [1, 1, step, 0.0, 0.001, 10, 1, 1, 1]
#         ub = [512, 5, step*3, 0.5, 0.2, 100, 1000, 5, 5]
#         #x0=neurons, x1=layers, x2=lookback
#         #x3=dropout, x4=learning rate, x5=epochs
#         #x6=batch_size, x7=number of lstm/conv1d layers, x8=pool_size
        

#         preds, infer_times = {}, {}
        
#         print("Step size: ", step)

#         DL_NAS.rpi_name = rpi
#         DL_NAS.model_name = f'{rpi}_{seq_type}_{step*5}sec'

#         DL_NAS.run_pso(step, lb, ub)