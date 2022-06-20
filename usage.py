# this file holds class to get usage statistics


from time import time, sleep, localtime
from os import system
from queue import LifoQueue
from threading  import Thread
import psutil 
from py3nvml import grab_gpus
import py3nvml.nvidia_smi
import pandas as pd
import subprocess
import re

pd.options.mode.chained_assignment = None  # default='warn'

#if sys.platform

class Usage():

    def __init__(self, interval = 2, IO_SCALE_FACTOR = 100_000_000):
    
        self.IO_SCALE_FACTOR = IO_SCALE_FACTOR
        self.q = LifoQueue()
        self.state, self.net = None, None
        self.kill_monitor_thread = False # unsets flag that kills monitoring thread
        self.get_current = False # unsets flag to place usage info in queue from thread
        if grab_gpus() == 0:
            self.gpu = False
        else:
            self.gpu = True

        self.interval = interval


    def run_glances(self):
        """Runs galnces in command line with a 15 second interval and exports a csv file containing the usage data"""
        print('starting glances')
        system('cmd /k "glances -t 15 --export csv --export-csv-file benchmark.csv"')


    def get_WiFi(self):
        
        iwconfig_output = str(subprocess.check_output(["iwconfig",'wlan0']))
        iwconfig_output = re.split('=| |:|\n',iwconfig_output)
        iwconfig_output = list(filter(None, iwconfig_output)) #TODO: check effect, might need to change below index shift values
        try:
            if iwconfig_output[iwconfig_output.index('Point') + 2] == 'Not-Associated':
                return 0, 0, 0, 0, 0
            else:
                wifi_freq = iwconfig_output[iwconfig_output.index('Frequency') + 1]
                bit_rate = iwconfig_output[iwconfig_output.index('Rate') + 1]
                lq = iwconfig_output[iwconfig_output.index('Quality') + 1].split('/')
                link_quality = lq[0]
                link_quality_max = lq[1]
                signal_level = iwconfig_output[iwconfig_output.index('level') + 1]
                return wifi_freq, bit_rate, link_quality, link_quality_max, signal_level
        except:
            return 0, 0, 0, 0, 0
        
    def get_full_usage(self):
        """Obtains one instance of usage information of system using the PsUtil Library at current time"""
        curr_time = time() # gets curretn time
        timeObj = localtime(curr_time)
        # creats time stamp Day-Month-Year Hour:Minute:Second
        time_stamp = str(timeObj.tm_mday) + '-' + str(timeObj.tm_mon) + '-' + str(timeObj.tm_year) + ' ' + str(timeObj.tm_hour) + ':' + str(timeObj.tm_min) + ':' + str(timeObj.tm_sec)
        
        cpu = psutil.cpu_percent(interval=1) # gets percentage use of cpu
        cpu_freq = psutil.cpu_freq().current # gets current cpu freq
        cpu_time = psutil.cpu_times() # get cpu state periods, eg. user, idle, system 
        net = psutil.net_io_counters() # gets network input and output information
        memory = psutil.virtual_memory()
        
        temp = psutil.sensors_temperatures()['cpu_thermal'][0].current
        
        wifi_freq, bit_rate, link_quality, link_quality_max, signal_level = self.get_WiFi()
         
        if self.gpu == True: # if gpu is availble obtains gpu information (usage and memory) 
            py3nvml.nvidia_smi.nvmlInit()
            handle = py3nvml.nvidia_smi.nvmlDeviceGetHandleByIndex(0)
            gpu_res = py3nvml.nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
            return time_stamp, curr_time, cpu_freq, cpu, cpu_time, net, memory, temp, gpu_res, wifi_freq, bit_rate, link_quality, link_quality_max, signal_level
        else: # if gpu is not availble returns 0
            return time_stamp, curr_time, cpu_freq, cpu, cpu_time, net, memory, temp, 0, wifi_freq, bit_rate, link_quality, link_quality_max, signal_level
    
    def get_ipykernel_usage_dict(self):
        """Obtains one instance of usage information using the PsUtil Library at current time for the processes related to jupyter notebook (ipykernel)"""
        proc = {p.pid: p.info for p in psutil.process_iter(['name','cmdline'])}
        proc_usage_dict = {}
        for i in proc:
            if proc[i]['name'] == 'python3':
                for x in proc[i]['cmdline']:
                    if x == 'ipykernel_launcher':
                        model_training_proc = psutil.Process(i)
                        break
                try:
                    model_training = model_training_proc.as_dict(attrs=['cpu_percent','memory_percent','cpu_times','memory_percent', 'io_counters'])
                    for key, value in model_training.items():
                        if type(value) != float:
                            value = value._asdict()
                            for k, val in value.items():
                                ky = '{}_{}_ipykernel'.format(key, k)
                                proc_usage_dict.update({ky:[val]})
                        else:
                            ky = '{}_ipykernel'.format(key)
                            proc_usage_dict.update({ky:[value]})
                    break
                except:
                    pass
        return proc_usage_dict
    
    def get_topProcesses_usage_dict(self):
        """Obtains one instance of usage information using the PsUtil Library at current time for process using resources the most"""
        procs = {p.pid: p.info for p in psutil.process_iter(['name','cpu_percent','memory_percent'])}
        proc_usage_dict = {}
        for i in procs:
            if procs[i]['cpu_percent'] >= 10 or procs[i]['memory_percent'] >= 3:
                proc_name = procs[i]['name'].split(' ')[0]
                
                if proc_name == 'chromium-browser-v7':
                    proc_name = '{}{}'.format(proc_name, procs[i]['name'].split(' ')[1])
                
                topProc = psutil.Process(i)
                topProc_dict = topProc.as_dict(attrs=['cpu_percent','cpu_times','memory_percent', 'io_counters'])
                try:
                    for key, value in topProc_dict.items():
                        if type(value) != float:
                            value = value._asdict()
                            for k, val in value.items():
                                ky = '{}_{}_{}'.format(key, k, proc_name)
                                proc_usage_dict.update({ky:[val]})
                        else:
                            ky = '{}_{}'.format(key, proc_name)
                            proc_usage_dict.update({ky:[value]})
                except:
                    pass
        return proc_usage_dict

    def get_usage_df(self, ipykernel, topProc):
        """Gets usage infomation and converts them into a pandas Dataframe"""
        time_stamp, current_time, cpu_freq, cpu, cpu_time, net, memory, temp, res, wifi_freq, bit_rate, link_quality, link_quality_max, signal_level = self.get_full_usage()
        
        # creats dictionary with useage information to be added to the dataframe
        usage_dict = {
                'time_stamp': [time_stamp], 'time': [current_time],
                'cpu_freq':[cpu_freq], 'cpu': [cpu], 'cpu_user_time':[cpu_time.user], 'cpu_system_time':[cpu_time.system],'cpu_idle_time': [cpu_time.idle],
                'memory':[memory.percent],
                'net_sent': [net.bytes_sent], 'net_recv': [net.bytes_recv],
                'net_upload_rate': [0], 'net_download_rate': [0],
                'temp': [temp],
                'wifi_freq':[wifi_freq], 'bit_rate':[bit_rate], 'link_quality':[link_quality], 'link_quality_max':[link_quality_max], 'signal_level':[signal_level]
                }
        
        if self.gpu == True:
            gpu_dict = {'gpu': [res.gpu], 'gpu_mem': [res.memory]}
        else:
            gpu_dict = {'gpu': [0], 'gpu_mem': [0]}
            
        usage_dict.update(gpu_dict)
        
        if ipykernel == True:
            proc_usage_dict = self.get_ipykernel_usage_dict()
            usage_dict.update(proc_usage_dict)

        if topProc == True:
            topProc_uasge_dict = self.get_topProcesses_usage_dict()
            usage_dict.update(topProc_uasge_dict)
               
        if self.state != None:
            usage_dict.update({'state':[self.state]})
        
        if self.net != None:
            usage_dict.update({'net':[self.net]})
        
        
        return pd.DataFrame(usage_dict).set_index('time_stamp') # returns dataframe with time stamp as index
  
    def get_net_rate(self, df):
        """Obtain upload and download rate based on the monitorng interval as the bytes sent|received / time spent sending|recieiving in MBytes/s """
        df['net_upload_rate'].iloc[-1] = ( df['net_sent'][len(df)-1] - df['net_sent'][len(df)-2] ) / ( ( df['time'][len(df)-1] - df['time'][len(df)-2] ) * self.IO_SCALE_FACTOR )
        df['net_download_rate'].iloc[-1]  = ( df['net_recv'][len(df)-1] - df['net_recv'][len(df)-2] ) / ( ( df['time'][len(df)-1] - df['time'][len(df)-2] ) * self.IO_SCALE_FACTOR )
  
    def run_monitor(self, ipykernel = False, topProc = False, interval=2):
        """Gets usage information according to the interval set, to be used in a seprate threat so as not to stop code flow, returns usage information as a Dataframe when stop_monitor_thread is set"""
        get_usage_time = 0
        usage_dataframe = self.get_usage_df(ipykernel, topProc) # initialize usage information dataframe
        while True:
            if self.kill_monitor_thread == True: # kill thread when stop thread flag is set externally
                self.q.put(usage_dataframe) # place dataframe in queue enabling access from outside of thread
                break
            elif self.get_current == True: # place latest usage info data in queue
                self.q.put(usage_dataframe) # place dataframe in queue enabling access from outside of thread

            if interval-get_usage_time > 0:
                sleep(interval-get_usage_time) # used to ensure interval timing is accurate, taking out time to obtain info
            
            start_time = time()
            usage_dataframe = pd.concat([usage_dataframe,self.get_usage_df(ipykernel, topProc)]) # update usage information dataframe
            self.get_net_rate(usage_dataframe)
            get_usage_time = time() - start_time


    def run_monitor_thread(self, ipykernel, topProc):
        """ Runs monitoring thread """
        self.kill_monitor_thread = False
        self.t1 = Thread(target=self.run_monitor, args=[ipykernel, topProc, self.interval]) # creates monitoring thread and passes queue to return results in, with specified info retrieval interval
        self.t1.daemon = True
        self.t1.start()

    def get_current_usage(self):
        """ returns dataframe with current worker usage information """
        self.get_current = True
        #sleep(self.interval) # required to give time for the loop in the run_monitor function to read the q.put() line
        while len(self.q.queue) == 0: # wait until queue not empty 
            pass
        
        usage_data = self.q.get()
        self.get_current = False
        return usage_data

    def stop_monitor_thread(self):    
        """ stops monitoring thread and returns last worker usage information dataframe """
        self.kill_monitor_thread = True

        while len(self.q.queue) == 0: # wait until queue not empty
            pass
        
        usage_data = self.q.get()

        self.t1.join()  # kills monitoring thread
        return usage_data # returns last usage information dataframe stored in queue


if __name__ == '__main__':

    monitor =  Usage(interval=5)
    
    monitor.run_monitor_thread(False,True)
    
    sleep(60)
    
    df = monitor.stop_monitor_thread()

    df.to_csv('monitor_trial.csv')