#%%
import os
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from statistics import mean
import json
#%%

sequences = ['pattern', 'random']
numbers = ['']#,'_3','_4']
step_sizes = [ '1Step', '2Step', '5Step', '10Step', '15Step', '30Step', '60Step'] 
periods = [5,10,25,50,75,150,300]
resources = ['cpu_user_time_diff','cpu_system_time_diff','cpu_idle_time_diff', 'memory']

# Determine the new file name
device_mapping = {
    "RPi4B8GB": "1800",
    "RPi4B4GB": "1500",
    "RPi4B2GB2": "1500",
    "RPi4B2GB1": "1200"
}

def save_metrics(data, model_type, step, res):

    def calculate_smape(actual, predicted):
        if not all([isinstance(actual, np.ndarray), isinstance(predicted, np.ndarray)]):
            actual, predicted = np.array(actual), np.array(predicted)
        return round(np.mean(np.abs(predicted - actual) / ((np.abs(predicted) + np.abs(actual)) / 2)) * 100, 2)

    metrics = {}



    for resource in res:
        preds = data[resource+'_preds'].to_numpy().flatten()
        actual = data[resource+'_obs'].to_numpy().flatten()

        score = round(r2_score(actual, preds), 3)
        rmse = round(mean_squared_error(actual, preds, squared=False), 3)
        mae = round(mean_absolute_error(actual, preds), 3)
        mape = round(mean_absolute_percentage_error(actual, preds) * 100, 2)
        smape = calculate_smape(actual, preds)

        metrics[resource] = {
            "Resource": resource,
            "Score": score,
            "RMSE": rmse,
            "MAE": mae,
            "MAPE": mape,
            "SMAPE": smape,
            "Training time": 0,
            "Mean Inference time": 0,
            "Max Inference time": 0,
            "Min Inference time": 0
        }

    metrics_df = pd.DataFrame.from_dict(metrics, orient='index')

    metrics_dir = os.path.join(parent_folder, 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)

    metrics_file = os.path.join(metrics_dir, f"{device_name}_{device_speed}MHz_{model_type}_{step}.csv")
    metrics_df.to_csv(metrics_file, index=False)
    print(f"Metrics saved to: {metrics_file}")


def conv_to_num(df):
    # Convert the relevant columns to float
    for col in ['cpu_user_time_diff_obs', 'cpu_system_time_diff_obs', 'cpu_idle_time_diff_obs', 'memory_obs',
                'cpu_user_time_diff_preds', 'cpu_system_time_diff_preds', 'cpu_idle_time_diff_preds', 'memory_preds']:
        df[col] = df[col].apply(lambda x: json.loads(x)[0] if isinstance(x, str) and x.startswith('[') else x)
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df.dropna(inplace=True)

    return df



for SEQ in sequences:
    for NUM in numbers:
        for p, s in zip(periods, step_sizes):
            # Read the original CSV file
            try:
                # csv_file = f"HSMM_Results_rvp_{SEQ}_48hr{NUM}_1500sec_lb_{p}sec_pw.csv"
                

                # Get all the csv files in the current directory
                csv_files = [f for f in os.listdir() if f.endswith(str(p) + 'sec.csv')]

                for csv_file in csv_files:
                    print(csv_file)

                    data = pd.read_csv(csv_file, error_bad_lines=False)

                    # Modify the content
                    data.columns
                    modified_data = data.rename(columns={
                        'cpu_user_time_diff_observations': 'cpu_user_time_diff_obs',
                        'cpu_user_time_diff_predicted_observations': 'cpu_user_time_diff_preds',
                        'cpu_system_time_diff_observations': 'cpu_system_time_diff_obs',
                        'cpu_system_time_diff_predicted_observations': 'cpu_system_time_diff_preds',
                        'cpu_idle_time_diff_observations': 'cpu_idle_time_diff_obs',
                        'cpu_idle_time_diff_predicted_observations': 'cpu_idle_time_diff_preds',
                        'memory_observations': 'memory_obs',
                        'memory_predicted_observations': 'memory_preds'
                    })

                    parent_folder = os.path.dirname(os.path.dirname(os.path.abspath(csv_file)))
                    device_name = os.path.basename(parent_folder)
                    device_speed = device_mapping.get(device_name, "unknown")

                    new_csv_file = f"{device_name}_{device_speed}MHz_{SEQ}{NUM}_{s}.csv"

                    # Save the modified content to the new CSV file
                    modified_data.to_csv(new_csv_file, index=False)
                    print(f"Modified CSV file saved as: {new_csv_file}")

                    modified_data = conv_to_num(modified_data)

                    save_metrics(data=modified_data,
                                model_type=SEQ,
                                step=s, 
                                res = resources)


            except FileNotFoundError:
                print(f"File not found: {csv_file}")
                continue

            


# %%
