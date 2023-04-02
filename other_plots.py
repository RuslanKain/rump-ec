
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from json import loads
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from statistics import mean
import matplotlib.dates as mdates
from scipy.ndimage.filters import uniform_filter1d


def plot_pred_obs(obs_data, pred_data1,pred_data2,pred_data3,                    
                  method1, method2, method3,
                  color1, color2, color3, title, unit, type, rpi, filter):
  
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.plot(obs_data,label = 'Observed', color = '#4876FF', linestyle = '-',alpha = 1)
  ax.plot(pred_data1,label = f'{method1}', color = color1,linestyle = '-',alpha = 0.7)
  ax.plot(pred_data2,label = f'{method2}', color = color2,linestyle = '--',alpha = 0.6)
  ax.plot(pred_data3,label = f'{method3}', color = color3,linestyle = '--',alpha = 0.5)

  print(title)
  # ax.set_title(method +' - '+ title)#, color = 'white')
  ax.grid(alpha=0.5)
  ax.legend(loc='best')
  #ax.tick_params(axis='x', colors='white')
  #ax.tick_params(axis='y', colors='white')
  #ax.yaxis.label.set_color('white')
  #ax.xaxis.label.set_color('white')
  ax.set_xlabel('Datapoint')#, color = 'white')
  ax.set_ylabel(unit)#, color = 'white')
  ax.set_ylim(bottom=0)
  plt.show()

# %%
"""Get prediction figures"""

rpi_name = 'RPi4B8GB_1800' #RPi4B8GB_1800, RPi4B4GB_1500, RPi4B2GB2_1500, RPi4B2GB1_1200
data_seq = 'pattern' # random, pattern
data_num = '' # _2, _3, _4
methods = ['HBPSHPO', 'HSMM','kNN-TSPI','HBLED']
methods_dict = {'HBPSHPO':{},'HSMM':{},'kNN-TSPI':{},'HBLED':{}}
lookahead_list = [1]#,2,5,10,15,30,60]
filter = 4 # 1, 25, 50, 100

for method in methods:
  for lookahead in lookahead_list:
    print(f"{method} figures for {rpi_name}_{data_seq}{data_num} for {lookahead} step")
    model_name = f'{rpi_name}_{data_seq}_{lookahead*5}sec'
    if method == 'HSMM':
      obs_dict = pd.read_csv(f"Results/{rpi_name.split('_')[0]}/predictions/HSMM_Results_rvp_{data_seq}_48hr{data_num}_1500sec_lb_{lookahead*5}sec_pw.csv")
    elif method == 'HBPSHPO':
      obs_dict = pd.read_csv(f"Results/{rpi_name.split('_')[0]}/predictions/HBPSHPO_Results_{model_name}.csv")
    elif method == 'kNN-TSPI' or method == 'HBLED':
      obs_dict = pd.read_csv(f"Results/{rpi_name.split('_')[0]}/predictions/{rpi_name.split('_')[0]}_{rpi_name.split('_')[1]}MHz_{data_seq}{data_num}_{lookahead}Step.csv")
    methods_dict[method]['user_cpu_obs'], methods_dict[method]['user_cpu_pred_obs'] = [], []
    methods_dict[method]['system_cpu_obs'], methods_dict[method]['system_cpu_pred_obs'] = [], []
    methods_dict[method]['idle_cpu_obs'], methods_dict[method]['idle_cpu_pred_obs'] = [], []
    methods_dict[method]['ram_obs'], methods_dict[method]['ram_pred_obs'] = [], []
    

    for i in range(0,len(obs_dict),lookahead):
      
      methods_dict[method]['user_cpu_obs'].extend(loads(obs_dict['cpu_user_time_diff_observations'][i]))
      methods_dict[method]['user_cpu_pred_obs'].extend(loads(obs_dict['cpu_user_time_diff_predicted_observations'][i]))
      methods_dict[method]['system_cpu_obs'].extend(loads(obs_dict['cpu_system_time_diff_observations'][i]))
      methods_dict[method]['system_cpu_pred_obs'].extend(loads(obs_dict['cpu_system_time_diff_predicted_observations'][i]))
      methods_dict[method]['idle_cpu_obs'].extend(loads(obs_dict['cpu_idle_time_diff_observations'][i]))
      methods_dict[method]['idle_cpu_pred_obs'].extend(loads(obs_dict['cpu_idle_time_diff_predicted_observations'][i]))
      methods_dict[method]['ram_obs'].extend(loads(obs_dict['memory_observations'][i]))
      methods_dict[method]['ram_pred_obs'].extend(loads(obs_dict['memory_predicted_observations'][i]))


shift = -300
# 'user_cpu_pred_obs', 'system_cpu_pred_obs', 'idle_cpu_pred_obs', 'ram_pred_obs'
# 'user_cpu_obs', 'system_cpu_obs', 'idle_cpu_obs', 'ram_obs'
#plot_pred_obs(pred_datas, obs_data, title, unit, type, rpi, methods, colors
plot_pred_obs(uniform_filter1d(methods_dict['HSMM']['user_cpu_pred_obs'][4000:8000], size=filter),uniform_filter1d(methods_dict['HBPSHPO']['user_cpu_pred_obs'][4000-shift:8000-shift], size=filter),uniform_filter1d(methods_dict['HSMM']['user_cpu_obs'][4000:8000], size=filter) , 'CPU User Time', 'Seconds', data_seq + data_num, rpi_name.split('_')[0], 'RUMP','HBPSHPO', filter)

# %%
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
  rmse_bars = ax.bar(rmse_xticks+0.14, [np.mean(r_y) for r_y in rmse_y], yerr= [np.std(r_y) for r_y in rmse_y], width=0.65, align='center', error_kw=dict(lw=4, capsize=8, capthick=3),alpha=0.95)
  mae_bars = ax.bar(mae_xticks-0.14, [np.mean(m_y) for m_y in mae_y], yerr= [np.std(m_y) for m_y in mae_y], width=0.65, align='center', error_kw=dict(lw=4, capsize=8, capthick=3),alpha=0.85)
  
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
"""Get error figures"""
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


  plt.rcParams.update({'font.size': 40})  
  fig = plt.figure(figsize=(24,24))
  ax = fig.add_subplot(1,1,1)
  print(res)
  ax.set_ylabel(res)
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
#Violin Plot
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
# %%