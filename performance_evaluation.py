#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, multilabel_confusion_matrix, classification_report
from statistics import mean
from sklearn.preprocessing import MinMaxScaler
from json import loads

from scipy.ndimage.filters import uniform_filter1d
import os
#%%
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

def plot_pred_obs_2(hsmm_pred_data,hbpshpo_pred_data, obs_data, title, unit, type, rpi, filter, save = False):

  fig = plt.figure(figsize=(10,5))
  ax = fig.add_subplot(111)

  ax.plot(obs_data,label = 'Observed', linewidth = 3, color = 'limegreen')
  ax.plot(hbpshpo_pred_data,label = 'HBPSHPO - Predicted', linewidth = 2, linestyle='--',  color = 'crimson')
  ax.plot(hsmm_pred_data,label = 'HSMM - Predicted', linewidth = 2, linestyle=':' , color = 'blue')
  
  ax.set_title('HSMM vs HBPHSPO - '+ title)
  ax.grid()
  ax.legend(loc='best')
  ax.set_xlabel('Datapoint')
  ax.set_ylabel(unit)
  if save == True:
    if filter == 1:
      f_path = 'Non Filtered'
    else:
      f_path = 'Filtered'
    plt_name = title.replace(" ", "_")
    try:
      os.makedirs(r"figures/{}/{}/{}/{} Step/{}".format(type,rpi,'HSMM vs HBPHSPO',lookahead,f_path))
    except:
      pass
    save_path = r"figures/{}/{}/{}/{} Step/{}/{}_{}step_{}ma.png".format(type,rpi,'HSMM vs HBPHSPO',lookahead,f_path,plt_name,lookahead,filter) 
    plt.savefig(save_path)
  else:
    plt.show()

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

def add_line(ax, xpos, ypos, plus, minus):
  line = plt.Line2D([xpos, xpos], [ypos + plus, ypos - minus],
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

  color_pair1 = ['#aaffc3', '#8B3A3A']  # color pair 1:  Mint, Apricot: #ffd8b1, Blueviolet: #8A2BE2
  color_pair2 = ['#458B74','#FF6A6A']  # color pair 2: Teal, Beige: #fffac8, Pink: #fabed4


  # define list of dark colors
  # https://sashat.me/2017/01/11/list-of-20-simple-distinct-colors/
  rmse_bars = ax.bar(rmse_xticks+0.14, [np.mean(r_y) for r_y in rmse_y], yerr= [np.std(r_y) for r_y in rmse_y], width=0.65, align='center', error_kw=dict(lw=4, capsize=8, capthick=3))
  mae_bars = ax.bar(mae_xticks-0.14, [np.mean(m_y) for m_y in mae_y], yerr= [np.std(m_y) for m_y in mae_y], width=0.65, align='center', error_kw=dict(lw=4, capsize=8, capthick=3))
  
  hatches = ['','///']*14

  count = 0

  for i in range(len(mae_bars)):
    count += 1
    if count == 1 or count == 2:
        mae_bars[0].set(label = 'MAE  - RUMP')
        mae_bars[i].set(hatch=hatches[i], facecolor=color_pair1[1]) # edgecolor='black'
        rmse_bars[0].set(label = 'RMSE  - RUMP')
        rmse_bars[i].set(hatch=hatches[i], facecolor=color_pair2[1])
    else:
        mae_bars[2].set(label = 'MAE  - HBPSHPO')
        mae_bars[i].set(hatch=hatches[i],  facecolor=color_pair2[0])
        rmse_bars[2].set(label = 'RMSE  - HBPSHPO')
        rmse_bars[i].set(hatch=hatches[i], facecolor=color_pair1[0])
        if count == 4:
           count = 0


  ax.set_xticks(mae_xticks)
  ax.set_xticklabels(mae_x)
  ax.set_xlim(.5, mae_ly + .5)
  ax.yaxis.grid(True)

  scale = 1. / mae_ly
  for pos in range(mae_ly + 1):
      add_line(ax, pos * scale, -.05, 0.05, 0)
  ypos = -.24
  while mae_groups:
      group = mae_groups.pop()
      pos = 0
      for label, rpos in group:
          lxpos = (pos + .5 * rpos) * scale
          if 'step' in label:
            ax.text(lxpos, ypos, label, ha='center', transform=ax.transAxes)
          else:
            ypos = -.24
            ax.text(lxpos, ypos, label, ha='center', transform=ax.transAxes, rotation=72)
          add_line(ax, pos * scale, ypos, .25, 0)
          pos += rpos
      add_line(ax, pos * scale, ypos, .25, 0)
      add_line(ax, pos * scale, ypos, .08, .02)
      ypos -= .06

# %%
"""Get RUMP or HBPSHPO prediction figures"""

rpi_name = 'RPi4B8GB_1800' #RPi4B8GB_1800, RPi4B4GB_1500, RPi4B2GB2_1500, RPi4B2GB1_1200
data_seq = 'pattern' # random, pattern
data_num = '' # _2, _3, _4
method = 'HBPSHPO' # HBPSHPO, HSMM
lookahead_list = [1, 2, 5, 10, 15, 30, 60]
filter = 25 # 1, 25, 50, 100

for lookahead in lookahead_list:
  print(f"{method} figures for {rpi_name}_{data_seq}{data_num} for {lookahead} step")
  model_name = f'{rpi_name}_{data_seq}_{lookahead*5}sec'
  if method == 'HSMM':
    obs_dict = pd.read_csv(f"Results/{rpi_name.split('_')[0]}/HSMM_Results_rvp_{data_seq}_48hr{data_num}_1500sec_lb_{lookahead*5}sec_pw.csv")
  elif method == 'HBPSHPO':
    obs_dict = pd.read_csv(f"Results/{rpi_name.split('_')[0]}/HBPSHPO_Results_{model_name}.csv")

  user_cpu_obs, user_cpu_pred_obs = [], []
  system_cpu_obs, system_cpu_pred_obs = [], []
  idle_cpu_obs, idle_cpu_pred_obs = [], []
  ram_obs, ram_pred_obs = [], []
  #label_obs, label_pred_obs = [], []
  #net_sent_obs, net_sent_pred_obs = [], []
  

  for i in range(0,len(obs_dict),lookahead):
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
    
  # change last arg to True to save figures
  plot_pred_obs(uniform_filter1d(user_cpu_pred_obs, size=filter),uniform_filter1d(user_cpu_obs, size=filter) , 'CPU User Time', 'Seconds', data_seq + data_num, rpi_name.split('_')[0], method, filter, False)
  plot_pred_obs(uniform_filter1d(system_cpu_pred_obs, size=filter), uniform_filter1d(system_cpu_obs, size=filter) , 'CPU System Time', 'Seconds', data_seq + data_num, rpi_name.split('_')[0], method, filter, False)
  plot_pred_obs(uniform_filter1d(idle_cpu_pred_obs, size=filter), uniform_filter1d(idle_cpu_obs, size=filter), 'CPU Idle Time', 'Seconds', data_seq + data_num, rpi_name.split('_')[0], method, filter, False)
  plot_pred_obs(uniform_filter1d(ram_pred_obs, size=filter), uniform_filter1d(ram_obs, size=filter), 'RAM', 'Memory (%)', data_seq + data_num, rpi_name.split('_')[0], method, filter, False)
  #plot_pred_obs(uniform_filter1d(net_sent_pred_obs, size=filter) , uniform_filter1d(net_sent_obs, size=filter), 'Network Upload', 'Bytes', method, filter, True)


# %%
"""Get RUMP vs HBPSHPO prediction figures"""
rpi_name = 'RPi4B8GB_1800' #RPi4B8GB_1800, RPi4B4GB_1500, RPi4B2GB2_1500, RPi4B2GB1_1200
data_seq = 'pattern' # random, pattern
data_num = '' # _2, _3, _4
methods = ['HBPSHPO', 'HSMM']
lookahead_list = [1]
filter = 10 # 1, 25, 50, 100

for lookahead in lookahead_list:
  for method in methods:
    print(f"{method} figures for {rpi_name}_{data_seq}{data_num} for {lookahead} step")
    model_name = f'{rpi_name}_{data_seq}_{lookahead*5}sec'
    if method == 'HSMM':
      hsmm_obs_dict = pd.read_csv(f"Results/{rpi_name.split('_')[0]}/HSMM_Results_rvp_{data_seq}_48hr{data_num}_1500sec_lb_{lookahead*5}sec_pw.csv")
    elif method == 'HBPSHPO':
      hbpshpo_obs_dict = pd.read_csv(f"Results/{rpi_name.split('_')[0]}/HBPSHPO_Results_{model_name}.csv")

    user_cpu_obs, hsmm_user_cpu_pred_obs, hbpshpo_user_cpu_pred_obs = [], [], []
    system_cpu_obs, hsmm_system_cpu_pred_obs, hbpshpo_system_cpu_pred_obs = [], [], []
    idle_cpu_obs, hsmm_idle_cpu_pred_obs, hbpshpo_idle_cpu_pred_obs = [], [], []
    ram_obs, hsmm_ram_pred_obs, hbpshpo_ram_pred_obs = [], [], []
  
  # required because of 300 datapoints required the hsmm model for lookback 
  shift = 300
  cut = 2

  for i in range(0,int((min(len(hsmm_obs_dict), len(hbpshpo_obs_dict))-shift)/cut),lookahead):
    user_cpu_obs.extend(loads(hsmm_obs_dict['cpu_user_time_diff_observations'][i]))
    hsmm_user_cpu_pred_obs.extend(loads(hsmm_obs_dict['cpu_user_time_diff_predicted_observations'][i]))
    hbpshpo_user_cpu_pred_obs.extend(loads(hbpshpo_obs_dict['cpu_user_time_diff_predicted_observations'][i+shift]))

    system_cpu_obs.extend(loads(hsmm_obs_dict['cpu_system_time_diff_observations'][i]))
    hsmm_system_cpu_pred_obs.extend(loads(hsmm_obs_dict['cpu_system_time_diff_predicted_observations'][i]))
    hbpshpo_system_cpu_pred_obs.extend(loads(hbpshpo_obs_dict['cpu_system_time_diff_predicted_observations'][i+shift]))

    idle_cpu_obs.extend(loads(hsmm_obs_dict['cpu_idle_time_diff_observations'][i]))
    hsmm_idle_cpu_pred_obs.extend(loads(hsmm_obs_dict['cpu_idle_time_diff_predicted_observations'][i]))
    hbpshpo_idle_cpu_pred_obs.extend(loads(hbpshpo_obs_dict['cpu_idle_time_diff_predicted_observations'][i+shift]))

    ram_obs.extend(loads(hsmm_obs_dict['memory_observations'][i]))
    hsmm_ram_pred_obs.extend(loads(hsmm_obs_dict['memory_predicted_observations'][i]))
    hbpshpo_ram_pred_obs.extend(loads(hbpshpo_obs_dict['memory_predicted_observations'][i+shift]))
    
  # change last arg to True to save figures
  plot_pred_obs_2(uniform_filter1d(hsmm_user_cpu_pred_obs, size=filter), uniform_filter1d(hbpshpo_user_cpu_pred_obs, size=filter),uniform_filter1d(user_cpu_obs, size=filter) , 'CPU User Time', 'Seconds', data_seq + data_num, rpi_name.split('_')[0], filter, False)
  plot_pred_obs_2(uniform_filter1d(hsmm_system_cpu_pred_obs, size=filter), uniform_filter1d(hbpshpo_system_cpu_pred_obs, size=filter), uniform_filter1d(system_cpu_obs, size=filter) , 'CPU System Time', 'Seconds', data_seq + data_num, rpi_name.split('_')[0], filter, False)
  plot_pred_obs_2(uniform_filter1d(hsmm_idle_cpu_pred_obs, size=filter), uniform_filter1d(hbpshpo_idle_cpu_pred_obs, size=filter), uniform_filter1d(idle_cpu_obs, size=filter), 'CPU Idle Time', 'Seconds', data_seq + data_num, rpi_name.split('_')[0], filter, False)
  plot_pred_obs_2(uniform_filter1d(hsmm_ram_pred_obs, size=filter), uniform_filter1d(hbpshpo_ram_pred_obs, size=filter), uniform_filter1d(ram_obs, size=filter), 'RAM', 'Memory (%)', data_seq + data_num, rpi_name.split('_')[0], filter, False)

# %%
"""Get RUMP vs HBPSHPO Error figures"""
rpi_name = 'RPi4B2GB1_1200' #RPi4B8GB_1800, RPi4B4GB_1500, RPi4B2GB2_1500, RPi4B2GB1_1200
rpi_name_list = ['RPi4B8GB_1800', 'RPi4B4GB_1500', 'RPi4B2GB2_1500', 'RPi4B2GB1_1200']
data_seq = 'random' # random, pattern
data_seq_list = ['random','pattern']
data_num = '' # _2, _3, _4

lookahead_list = [1,2,5,10,15,30,60]
HSMM_error_list, HBPSHPO_error_list = [], []


HSMM_columns = ['cpu_user_time_diff_mae','cpu_user_time_diff_rmse','cpu_system_time_diff_mae','cpu_system_time_diff_rmse','cpu_idle_time_diff_mae','cpu_idle_time_diff_rmse','memory_mae','memory_rmse']
HBPSHPO_columns = ['User CPU mae','User CPU rmse','System CPU mae','System CPU rmse','Idle CPU mae','Idle CPU rmse','RAM mae','RAM rmse']
Resources = {'User CPU Time (sec.)':['cpu_user_time_diff_mae','cpu_user_time_diff_rmse','User CPU mae','User CPU rmse'],
             'System CPU Time (sec.)':['cpu_system_time_diff_mae','cpu_system_time_diff_rmse','System CPU mae','System CPU rmse'],
             'Idle CPU Time (sec.)':['cpu_idle_time_diff_mae','cpu_idle_time_diff_rmse', 'Idle CPU mae','Idle CPU rmse'],
             'Memory Percent Usage':['memory_mae','memory_rmse','RAM mae','RAM rmse']}

mae_data_dict_full, rmse_data_dict_full = {}, {}

for res in Resources:
  mae_data_dict, rmse_data_dict = {}, {}
  

  for lookahead in lookahead_list:

    mae_data_dict[f"{lookahead} step"] = {'RUMP': {'R':[],'P':[]}, 'HBPSHPO':{'R':[],'P':[]}}
    rmse_data_dict[f"{lookahead} step"] = {'RUMP': {'R':[],'P':[]}, 'HBPSHPO':{'R':[],'P':[]}}

    for rpi_name in rpi_name_list:

      for data_seq in data_seq_list:

        

        #print(f"Error figures for {rpi_name}_{data_seq}{data_num} for {lookahead} step")

        model_name = f'{rpi_name}_{data_seq}_{lookahead*5}sec'

        HSMM_temp = pd.read_csv(f"Results/{rpi_name.split('_')[0]}/HSMM_Error_Results_rvp_{data_seq}_48hr{data_num}_1500sec_lb_{lookahead*5}sec_pw.csv")
        HBPSHPO_temp = pd.read_csv(f"Results/{rpi_name.split('_')[0]}/HBPSHPO_Error_Results_{model_name}.csv")

        if data_seq == 'random':
          data_seq = 'R'
        elif data_seq == 'pattern':
          data_seq = 'P'

        mae_data_dict[f"{lookahead} step"]['RUMP'][data_seq].append(HSMM_temp[Resources[res][0]][0])
        mae_data_dict[f"{lookahead} step"]['HBPSHPO'][data_seq].append(HBPSHPO_temp[Resources[res][2]][0])
        rmse_data_dict[f"{lookahead} step"]['RUMP'][data_seq].append(HSMM_temp[Resources[res][1]][0])
        rmse_data_dict[f"{lookahead} step"]['HBPSHPO'][data_seq].append(HBPSHPO_temp[Resources[res][3]][0])


      #mae_data_dict[f"{lookahead} step"]['HDP-HSMM'] = {k:float(sum(v))/len(v) for k, v in mae_data_dict[f"{lookahead} step"]['HDP-HSMM'].items()}
      #mae_data_dict[f"{lookahead} step"]['HBPSHPO'] = {k:float(sum(v))/len(v) for k, v in mae_data_dict[f"{lookahead} step"]['HBPSHPO'].items()}
      #rmse_data_dict[f"{lookahead} step"]['HDP-HSMM'] = {k:float(sum(v))/len(v) for k, v in rmse_data_dict[f"{lookahead} step"]['HDP-HSMM'].items()}
      #rmse_data_dict[f"{lookahead} step"]['HBPSHPO'] = {k:float(sum(v))/len(v) for k, v in rmse_data_dict[f"{lookahead} step"]['HBPSHPO'].items()}

  plt.rcParams.update({'font.size': 40})  
  fig = plt.figure(figsize=(24,24))
  ax = fig.add_subplot(1,1,1)
  ax.set_title(res)
  label_group_bar(ax, mae_data_dict, rmse_data_dict)
  plt.legend(loc='best')
  plt.grid(lw=3,ls='--', c='grey', axis='y')
  fig.subplots_adjust(bottom=0.3)
  
  if res == 'User CPU Time (sec.)':
    plt.yticks(np.arange(0, 8, 1))
  elif res == 'System CPU Time (sec.)':
    plt.yticks(np.arange(0, 0.7, 0.1))
  elif res == 'Idle CPU Time (sec.)':
    plt.yticks(np.arange(0, 8, 1))
  elif res == 'Memory Percent Usage':
    plt.yticks(np.arange(0, 11, 1))
  fig.savefig(r"figures/{} Error Plot.png".format(res))
  plt.show()

  mae_data_dict_full[res] = mae_data_dict
  rmse_data_dict_full[res] = rmse_data_dict

# %%

r_mae_percent, r_rmse_percent, p_mae_percent, p_rmse_percent = [], [], [], []
for (res1, mae_data), (res2, rmse_data) in zip(mae_data_dict_full.items(), rmse_data_dict_full.items()):
    for lookahead in lookahead_list:
        for data_type in ['R', 'P']:
            if data_type == 'R':
                r_mae_percent.append(100 * (np.mean(mae_data[f"{lookahead} step"]['RUMP'][data_type]) - np.mean(
                    mae_data[f"{lookahead} step"]['HBPSHPO'][data_type])) / np.mean(
                    mae_data[f"{lookahead} step"]['HBPSHPO'][data_type]))
                r_rmse_percent.append(100 * (np.mean(rmse_data[f"{lookahead} step"]['RUMP'][data_type]) - np.mean(
                    rmse_data[f"{lookahead} step"]['HBPSHPO'][data_type])) / np.mean(
                    rmse_data[f"{lookahead} step"]['HBPSHPO'][data_type]))
            elif data_type == 'P':
                p_mae_percent.append(100 * (np.mean(mae_data[f"{lookahead} step"]['RUMP'][data_type]) - np.mean(
                    mae_data[f"{lookahead} step"]['HBPSHPO'][data_type])) / np.mean(
                    mae_data[f"{lookahead} step"]['HBPSHPO'][data_type]))
                p_rmse_percent.append(100 * (np.mean(rmse_data[f"{lookahead} step"]['RUMP'][data_type]) - np.mean(
                    rmse_data[f"{lookahead} step"]['HBPSHPO'][data_type])) / np.mean(
                    rmse_data[f"{lookahead} step"]['HBPSHPO'][data_type]))

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111)
plt.rcParams.update({'font.size': 18})

boxprops = dict(linewidth=4)
medianprops = dict(linewidth=5, color='#458B74')
meanlineprops = dict(linewidth=5, color='#FF6A6A')
whiskerprops = dict(linewidth=4)
capprops = dict(linewidth=4)

box_plot_data = [r_mae_percent, p_mae_percent, r_rmse_percent, p_rmse_percent]
box_plot_colors = ['#aaffc3', '#8B3A3A', '#458B74', '#FF6A6A']

bp = plt.boxplot(box_plot_data, showmeans=True, showfliers=False, meanline=True, patch_artist=True,
                 medianprops=medianprops, meanprops=meanlineprops, boxprops=boxprops, whiskerprops=whiskerprops,
                 capprops=capprops)

for patch, color in zip(bp['boxes'], box_plot_colors):
    patch.set_facecolor("#FFF8DC")

ax.set_xticklabels(['Random MAE', 'Patterned MAE', 'Random RMSE', 'Patterned RMSE'])
plt.ylabel('Percent Difference')
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.yticks(np.arange(-30, 175, 20))


# Add legend for median and mean lines
mean_line = plt.Line2D([], [], color='#FF6A6A', linestyle='--', linewidth=5, label='Mean')
median_line = plt.Line2D([], [], color='#458B74', linestyle='-', linewidth=5, label='Median')
plt.legend(handles=[mean_line, median_line], loc='upper left', fontsize=18)

plt.show()

# %%
"""Plot RUMP vs HBPSHPO Error difference Violin Plots"""
r_mae_percent, r_rmse_percent, p_mae_percent, p_rmse_percent = [], [], [], []
for (res1, mae_data), (res2, rmse_data) in zip(mae_data_dict_full.items(), rmse_data_dict_full.items()):
    for lookahead in lookahead_list:
        for data_type in ['R', 'P']:
            if data_type == 'R':
                r_mae_percent.append(100 * (np.mean(mae_data[f"{lookahead} step"]['RUMP'][data_type]) - np.mean(
                    mae_data[f"{lookahead} step"]['HBPSHPO'][data_type])) / np.mean(
                    mae_data[f"{lookahead} step"]['HBPSHPO'][data_type]))
                r_rmse_percent.append(100 * (np.mean(rmse_data[f"{lookahead} step"]['RUMP'][data_type]) - np.mean(
                    rmse_data[f"{lookahead} step"]['HBPSHPO'][data_type])) / np.mean(
                    rmse_data[f"{lookahead} step"]['HBPSHPO'][data_type]))
            elif data_type == 'P':
                p_mae_percent.append(100 * (np.mean(mae_data[f"{lookahead} step"]['RUMP'][data_type]) - np.mean(
                    mae_data[f"{lookahead} step"]['HBPSHPO'][data_type])) / np.mean(
                    mae_data[f"{lookahead} step"]['HBPSHPO'][data_type]))
                p_rmse_percent.append(100 * (np.mean(rmse_data[f"{lookahead} step"]['RUMP'][data_type]) - np.mean(
                    rmse_data[f"{lookahead} step"]['HBPSHPO'][data_type])) / np.mean(
                    rmse_data[f"{lookahead} step"]['HBPSHPO'][data_type]))

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111)
plt.rcParams.update({'font.size': 18})

violin_plot_data = [r_mae_percent, p_mae_percent, r_rmse_percent, p_rmse_percent]
violin_plot_colors = ['#aaffc3', '#8B3A3A', '#458B74', '#FF6A6A']

vp = ax.violinplot(violin_plot_data)

for i, body in enumerate(vp['bodies']):
    body.set_facecolor(violin_plot_colors[i])
    body.set_edgecolor('black')
    body.set_linewidth(2)
    body.set_alpha(0.5)
    #change 

# Add mean and median lines
for i, data in enumerate(violin_plot_data):
    x = i + 1
    mean = np.mean(data)
    median = np.median(data)
    ax.plot([x - 0.15, x + 0.15], [mean, mean], color='#8B0A50', linestyle='-', linewidth=3)
    ax.plot([x - 0.15, x + 0.15], [median, median], color='#2F4F4F', linestyle='-', linewidth=3)

# Customize quartile lines
for i in range(len(violin_plot_data)):
    quartiles = np.percentile(violin_plot_data[i], [0.001, 99.999])
    ax.vlines(i + 1, quartiles[0], quartiles[1], color='black', linewidth=2)
    ax.hlines(quartiles, i + 1 - 0.2, i + 1 + 0.2, color='black', linewidth=2)

ax.set_xticks([1, 2, 3, 4])
ax.set_xticklabels(['Random MAE', 'Patterned MAE', 'Random RMSE', 'Patterned RMSE'])
plt.ylabel('Percent Difference')
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.yticks(np.arange(-30, 200, 20))

# Add annotations for the first violin chart
hbposhpo_annotation = 'HBPSHPO'
rump_annotation = 'RUMP'
first_violin_max = np.max(r_mae_percent)
first_violin_min = np.min(r_mae_percent)

ax.annotate(hbposhpo_annotation, xy=(1, first_violin_max), xytext=(1.15, first_violin_max + 7),
            arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=7), fontsize=18)

ax.annotate(rump_annotation, xy=(1, first_violin_min), xytext=(1.15, first_violin_min - 9),
            arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=7), fontsize=18)


# Add legend for median and mean lines
mean_line = plt.Line2D([], [], color='#8B0A50', linestyle='-', linewidth=3, label='Mean')
median_line = plt.Line2D([], [], color='#2F4F4F', linestyle='-', linewidth=3, label='Median')
plt.legend(handles=[mean_line, median_line], loc='upper left', fontsize=18)



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
        data_associations_dict[device]['train_data'][model] = (pd.read_csv(f"data/{device.split('_')[0]}/{device}_{data_associations_dict[device]['freq']}MHz_res_usage_data_train_pred_{model}.csv"))
        labeled_data_test = pd.read_csv(f"data/{device.split('_')[0]}/{device}_{data_associations_dict[device]['freq']}MHz_res_usage_data_test_pred_{model}.csv")
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
            data_associations_dict[device]['test_data'][model][f'accuracy - {prediction_window} step'] = accuracy.copy()
            mean_accuracy = round(np.mean(accuracy),2)
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
    plt.axhline(np.mean(train_states_random_1)*100,label='random - train')
    plt.axhline(np.mean(train_states_pattern_1)*100,color='orange',label='pattern - train')
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

# %%

def get_label(old_label):

    if old_label == 'random':
        return'Random'
    elif old_label == 'pattern':
        return'Pattern'
    elif old_label == 'RPi4B8GB':
        return'A'
    elif old_label == 'RPi4B4GB':
        return'B'
    elif old_label == 'RPi4B2GB2':
        return'C'
    elif old_label == 'RPi4B2GB1':
        return'D'


def get_color(dev):
    # '#aaffc3', '#8B3A3A', '#458B74', '#FF6A6A'
    if dev == 'RPi4B8GB': 
        return '#aaffc3'
    elif dev == 'RPi4B4GB':
        return '#8B3A3A'
    elif dev == 'RPi4B2GB2':
        return '#458B74'
    elif dev == 'RPi4B2GB1':    
        return '#FF6A6A'
    

import matplotlib.pyplot as plt

prediction_windows = [1,2,5,10,15,30,60]
sequence_type = ['random', 'pattern']
sq_test_accuracies, sq_train_accuracies = [], []
for sq_type in sequence_type:
    
    fig, ax = plt.subplots(figsize =(8, 6))
    plt.rcParams.update({'font.size': 14})
    test_accuracyies_list, train_accuracyies_list = [], []
    for device in test_accuracy:
        
        
        
        for model in test_accuracy[device]:
            if sq_type in model and '2' not in model:
                test_1 = test_accuracy[device][model]
                test_states_1 = test_accuracy_states[device][model]
                train_states_1 = train_accuracy_states[device][model]
            elif sq_type in model and '2' in model and '2GB1' not in device:
                test_2 = test_accuracy[device][model]
                test_states_2 = test_accuracy_states[device][model]
                train_states_2 = train_accuracy_states[device][model]
            elif sq_type in model and '3' in model and '2GB1' in device:
                test_2 = test_accuracy[device][model]
                test_states_2 = test_accuracy_states[device][model]
                train_states_2 = train_accuracy_states[device][model]

        print(f'{device}_{data_associations_dict[device]["freq"]}')

        
        #print('Avg')
        
        plt.plot(prediction_windows,[(t1 + t2) / 2 for t1, t2 in zip(test_1, test_2)], marker='o', markersize = 8, color = get_color(device) , linewidth = 2, linestyle = '--', label = f'{get_label(device)}  - Test')
        test_accuracyies_list.append([(t1 + t2) / 2 for t1, t2 in zip(test_1, test_2)])
        plt.axhline(np.mean([(ts1 + ts2) / 2 for ts1, ts2 in zip(train_states_1, train_states_2)])*100, linewidth = 2, color = get_color(device),label= f'{get_label(device)} - Train')
        train_accuracyies_list.append(np.mean([(ts1 + ts2) / 2 for ts1, ts2 in zip(train_states_1, train_states_2)])*100)

    plt.legend(loc='center right')
    plt.ylabel('Percent Accuracy')
    plt.xlabel('Steps')
    #plt.title(get_label(sq_type) + ' State Sequence')
    plt.grid()
    plt.yticks(np.arange(50, 105, 5))
    plt.xticks(prediction_windows)
    plt.setp(ax.xaxis.get_majorticklabels()[0],ha="right" )
    plt.setp(ax.xaxis.get_majorticklabels()[1:],ha="left" )
    
    plt.show() 
    print(sq_type, ' average train accuracy')
    print(sum(train_accuracyies_list) / len(train_accuracyies_list))
    sq_train_accuracies.append(sum(train_accuracyies_list) / len(train_accuracyies_list))

    print(sq_type, ' average test accuracy')
    print([sum(acc_sub_list) / len(acc_sub_list) for acc_sub_list in zip(*test_accuracyies_list)])
    sq_test_accuracies.append([sum(acc_sub_list) / len(acc_sub_list) for acc_sub_list in zip(*test_accuracyies_list)])
    """
    print('1')
    plt.plot(prediction_windows,test_1, marker='o', markersize = 8 , linewidth = 3, color ='limegreen', label = f'{get_label(device)} - {get_label(sq_type)} - Test')
    plt.axhline(mean(train_states_1)*100, linewidth = 3, color ='orange',label= f'{get_label(device)} - {get_label(sq_type)} - Train')
    plt.legend(loc='upper right')
    plt.ylabel('Percent Accuracy')
    plt.xlabel('Steps')
    plt.grid()
    plt.yticks(np.arange(50, 105, 5))
    plt.xticks(prediction_windows)
    plt.show() 

    print('2')
    plt.plot(prediction_windows,test_2, linewidth = 3, color ='orange',label= f'{get_label(device)} - {get_label(sq_type)} - Train')
    plt.axhline(mean(train_states_1)*100, linewidth = 3, color ='orange',label= f'{get_label(device)} - {get_label(sq_type)} - Train')
    plt.legend(loc='upper right')
    plt.ylabel('Percent Accuracy')
    plt.xlabel('Steps')
    plt.grid()
    plt.xticks(prediction_windows)
    plt.yticks(np.arange(50, 105, 5))
    plt.show()

    """


print('overall average train accuracy')
print(sum(sq_train_accuracies) / len(sq_train_accuracies))
   

print('overall average test accuracy')
print([sum(acc_sub_list) / len(acc_sub_list) for acc_sub_list in zip(*sq_test_accuracies)])
   
# %%
