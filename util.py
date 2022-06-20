import pandas as pd
from statistics import mean

 
def df_col_diff(df, columns):
        """Places the difference betweeen to rows of dataframe in a new column named with '_diff' """
        for i in range(len(columns)): 
            col = columns[i]
            name = col+'_diff'
            df[name] = df[col].diff()
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


"""For Regular Data"""
def align_df(QOS_df, usage_df):
    time_diff = []
    shift = len(QOS_df[QOS_df['QOS'] == 0])

    if QOS_df['QOS'][shift] > 5 * QOS_df['QOS'][shift:].mean():
        QOS_df = QOS_df[shift+1:]
    else:
        QOS_df = QOS_df[shift:]

    usage_df_temp = usage_df[shift:]
    usage_df_temp = usage_df_temp[:len(QOS_df)]
    min_len = min(len(QOS_df),len(usage_df_temp))
    
    for i in range(min_len-1): 
        diff = QOS_df['time'].iloc[i] - usage_df_temp['time'].iloc[i]
        time_diff.append(diff)
    
    if mean(time_diff) < -30:
        usage_df = usage_df[shift-7:]
    elif -30 <= mean(time_diff) < -25:
        usage_df = usage_df[shift-6:]
    elif -25 <= mean(time_diff) < -20:
        usage_df = usage_df[shift-5:]
    elif -20 <= mean(time_diff) < -15:
        usage_df = usage_df[shift-4:]
    elif -15 <= mean(time_diff) < -10:
        usage_df = usage_df[shift-3:]
    elif -10 <= mean(time_diff) < -5:
        usage_df = usage_df[shift-2:]
    elif -5 <= mean(time_diff) < -2.5:
        usage_df = usage_df[shift-1:]
    elif -2.5 <= mean(time_diff) <= 2.5:
        usage_df = usage_df[shift:]
    elif 2.5 < mean(time_diff) <= 5:
        usage_df = usage_df[shift+1:]
    elif 5 < mean(time_diff) <= 10:
        usage_df = usage_df[shift+2:]
    elif 10 < mean(time_diff) <= 15:
        usage_df = usage_df[shift+3:]
    elif 15 < mean(time_diff) <= 20:
        usage_df = usage_df[shift+4:]
    elif 20 < mean(time_diff) <= 25:
        usage_df = usage_df[shift+5:]
    elif 25 < mean(time_diff) <= 30:
        usage_df = usage_df[shift+6:]
    else:
        usage_df = usage_df[shift+7:]

    return QOS_df, usage_df[:len(QOS_df)]

def dataPrep(rpi_name, data_name, cols):
    usage_data = pd.read_csv("C:/Users/USER/Desktop/Queen's University/Research/rpi_benchmark_profile/data/{}/usage_data_{}_{}.csv".format(rpi_name,rpi_name,data_name), index_col = 'time_stamp')
    QOS_data = pd.read_csv("C:/Users/USER/Desktop/Queen's University/Research/rpi_benchmark_profile/data/{}/QOS_data_{}_{}.csv".format(rpi_name,rpi_name,data_name), index_col = 'time_stamp')
    RCoin = pd.read_csv("C:/Users/USER/Desktop/Queen's University/Research/rpi_benchmark_profile/data/{}/RCoin_{}_{}.csv".format(rpi_name,rpi_name,data_name))

    QOS_data, usage_data = align_df(QOS_data, usage_data)

    usage_data = df_col_diff(usage_data, cols)
    usage_data = usage_data.fillna(0)

    for index, row in RCoin.iterrows():
        if  isinstance(row['Training Losses'], float):
            RCoin.at[index,'Training Loss'] = row['Training Losses']
        else:
            x = row['Training Losses'].split(' ')[-1][:-1]
            RCoin.at[index,'Training Loss'] = x
            

    usage_data['cpu_times_ipykernel_percent'] = (usage_data['cpu_times_user_ipykernel_diff']+usage_data['cpu_times_system_ipykernel_diff'])/(usage_data['cpu_user_time_diff']+usage_data['cpu_system_time_diff'])

    return usage_data, QOS_data, RCoin

def sequenceData(data_list,sequence,durations):

    if len(sequence) != len(durations):
        print("Sequence and Durations lists' lengths must match")
        return 0

    #start, end = 0, 1
    df = pd.DataFrame({})

    for seq in sequence:

        indx = sequence.index(seq)
        end = durations[indx]
        
        df_add = data_list[seq].iloc[:end+1]
        data_list[seq] = data_list[seq].iloc[end+1:]

        df = pd.concat([df,df_add])

        #start = end

    return df
