#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, multilabel_confusion_matrix, classification_report
from statistics import mean
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from json import loads
#%%

def plot_states(colors):

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

def plot_accuracy_likelihood(prediction_window, rolling_window):

    print(f"Steps {prediction_window} - MA Window {rolling_window}")
    scaler = MinMaxScaler() 
    accuracy_scaled = scaler.fit_transform(labeled_data_test[f'accuracy - {prediction_window} step'].rolling(rolling_window).mean().values.reshape(-1, 1))
    log_likelihood_scaled = scaler.fit_transform(labeled_data_test[f'log_likelihood - {prediction_window} step'].values.reshape(-1, 1))

    plt.figure(figsize=(12, 8), dpi=80)
    plt.plot(accuracy_scaled, color='black', label='Accuracy')
    plt.plot(log_likelihood_scaled, color='b', label='Log Likelihood')
    
    colors = {'game': 'salmon', 'augmented_reality': 'lightblue', 'idle': 'plum', 'mining': 'peachpuff', 'stream': 'silver'}
    print(colors)
    plot_states(colors)

    plt.legend(loc='best')
    plt.grid()
    plt.show()


#%%


feature_names = 'cpu-all_mem'

model_names = ['rvp_random_48hr','rvp_pattern_48hr','rvp_random_48hr_2','rvp_pattern_48hr_2']

data_associations_dict = {'RPi4B8GB':{'freq':1800,'train_data':{},'test_data':{},'features':feature_names}, 'RPi4B4GB':{'freq':1500,'train_data':{},'test_data':{},'features':feature_names}, 'RPi4B2GB2':{'freq':1500,'train_data':{},'test_data':{},'features':feature_names}, 'RPi4B2GB1':{'freq':1200,'train_data':{},'test_data':{},'features':feature_names}} 

lookback = 300

#%%
"""Prepare data dict"""
for device in data_associations_dict:
    for model in model_names:
        data_associations_dict[device]['train_data'][model] = (pd.read_csv(f"data/{device}_{data_associations_dict[device]['freq']}MHz_res_usage_data_train_pred_{model}.csv"))
        labeled_data_test = pd.read_csv(f"data/{device}_{data_associations_dict[device]['freq']}MHz_res_usage_data_test_pred_{model}.csv")
        data_associations_dict[device]['test_data'][model] = (labeled_data_test[lookback:]) # remove lookback section
#%%
"""Training Classification Accuracy"""
train_accuracy_states = {}

for device in data_associations_dict:
    print(device)
    train_accuracy_states[device] = {}
    for model in data_associations_dict[device]['train_data']:
        print(model)
        train_accuracy_states[device][model] = []
        for name,group in data_associations_dict[device]['train_data'][model].groupby('state'):
            print(name)
            label = data_associations_dict[device]['train_data'][model].groupby('state').get_group(name)['predicted'].value_counts(normalize=True)
            print(label)
            train_accuracy_states[device][model].append(label.max())

#%%
"""Testing Classification Accuracy"""
test_accuracy_states = {}
for device in data_associations_dict:
    print(device)
    test_accuracy_states[device] = {}
    for model in data_associations_dict[device]['test_data']:
        print(model)
        test_accuracy_states[device][model] = []
        for name,group in data_associations_dict[device]['test_data'][model].groupby('state'):
            print(name)
            label = data_associations_dict[device]['test_data'][model].groupby('state').get_group(name)[f'predicted states - {1} step'].value_counts(normalize=True)
            test_accuracy_states[device][model].append(label.max())
            print(label)

# %%
"""Analysis of Predictions"""
prediction_windows = [1,2,5,10,15,30,60]

test_accuracy = {}

for device in data_associations_dict:
    print(device)
    test_accuracy[device] = {}
    for model in data_associations_dict[device]['test_data']:
        print(model)
        test_accuracy[device][model] = []
        for prediction_window in prediction_windows:
            accuracy = []

            for index, row in data_associations_dict[device]['test_data'][model].iterrows():
                if type(row[f'label - {prediction_window} step']) == str:
                    accuracy.append(accuracy_score(loads(row[f'label - {prediction_window} step']), loads(row[f'predicted states - {prediction_window} step']))*100)
                else:
                    accuracy.append(accuracy_score(row[f'label - {prediction_window} step'], row[f'predicted states - {prediction_window} step'])*100)
            data_associations_dict[device]['test_data'][model][f'accuracy - {prediction_window} step'] = accuracy
            mean_accuracy = round(mean(accuracy),2)
            test_accuracy[device][model].append(mean_accuracy)
            print(f"{prediction_window} step prediction accuracy: {mean_accuracy}%")
            #plot_accuracy_likelihood(prediction_window, 100)

# %%
import matplotlib.pyplot as plt


for device in test_accuracy:
    for model in test_accuracy[device]:
        if 'random' in model and '2' not in model:
            test_random_1 = test_accuracy[device][model]
            test_states_random_1 = test_accuracy_states[device][model]
            train_states_random_1 = train_accuracy_states[device][model]
        elif 'pattern' in model and '2' not in model:
            test_pattern_1 = test_accuracy[device][model]
            test_states_pattern_1 = test_accuracy_states[device][model]
            train_states_pattern_1 = train_accuracy_states[device][model]
        elif 'random' in model and '2' in model:
            test_random_2 = test_accuracy[device][model]
            test_states_random_2 = test_accuracy_states[device][model]
            train_states_random_2 = train_accuracy_states[device][model]
        elif 'pattern' in model and '2' in model:
            test_pattern_2 = test_accuracy[device][model]
            test_states_pattern_2 = test_accuracy_states[device][model]
            train_states_pattern_2 = train_accuracy_states[device][model]

    print(f'{device}_{data_associations_dict[device]["freq"]}')
   

    plt.plot(prediction_windows,[(r1 + r2) / 2 for r1, r2 in zip(test_random_1, test_random_2)], marker='o',label = 'random - test')
    plt.plot(prediction_windows,[(p1 + p2) / 2 for p1, p2 in zip(test_pattern_1, test_pattern_2)], marker='o',label = 'pattern - test')
    plt.axhline(mean(train_states_random_1)*100,label='random - train')
    plt.axhline(mean(train_states_pattern_1)*100,color='orange',label='pattern - train')
    plt.legend(loc='best')
    plt.ylabel('% Accuracy')
    plt.xlabel('Steps')
    plt.grid()
    plt.show() 
    
    labels = ['AR', 'Game', 'Idle', 'Mining', 'Stream']

    x = np.arange(len(labels))  # the label locations
    width = 0.25  # the width of the bars
    
    print( f'{device}_{data_associations_dict[device]["freq"]}')
    
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width*1.5, [round((r1 + r2) / 2,2)*100 for r1, r2 in zip(train_states_random_1, train_states_random_2)], width, label='random - train')
    rects3 = ax.bar(x + width*.5, [round((r1 + r2) / 2,2)*100 for r1, r2 in zip(test_states_random_1, test_states_random_2)], width, label='random - test')
    
    if '2GB1' in device:
        
        rects2 = ax.bar(x - width*.5, [ round(p*100,1) for p in train_states_pattern_1], width, label='pattern - train')
        test_states_pattern_1[1] = test_states_pattern_2[1]
        rects4 = ax.bar(x + width*1.5, [ round(r*100,1) for r in test_states_pattern_1], width, label='pattern - test')   
    else:
        rects2 = ax.bar(x - width*.5, [round((p1 + p2) / 2,3)*100 for p1, p2 in zip(train_states_pattern_1, train_states_pattern_2)], width, label='pattern - train')
        rects4 = ax.bar(x + width*1.5, [round((p1 + p2) / 2,3)*100 for p1, p2 in zip(test_states_pattern_1, test_states_pattern_2)], width, label='pattern - test')

    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('% Accuracy')
    ax.set_xlabel('States')
    ax.set_xticks(x,labels)
    ax.legend(loc='lower right')

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)
    ax.bar_label(rects3, padding=3)
    ax.bar_label(rects4, padding=3)

    #fig.tight_layout()

    plt.show()
# %%
