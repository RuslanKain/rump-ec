from time import sleep, time, strftime, localtime
from random import getrandbits, choice, seed, shuffle


from pynput.keyboard import Key, Controller # keyboard control
import imutils # basic image operations, such as resizing
import cv2 # computer vision lib


#required for application/state automation
from subprocess import Popen, check_call, run
from signal import SIGTERM
import os
import platform
from psutil import cpu_freq


#required for usage monitoring and frequent saving
from usage import Usage
from threading  import Thread

#required for AR
import numpy as np
import sys


def start():
    """Configures dynamic usage scenario"""
    interval = int(input("please enter monitoring interval size (>2 sec): "))

    while interval < 2:
        print("minimum monitoring interval size is 2 sec.")
        chunk_size = int(input("please enter monitoring interval size: "))

    cpuFreq = int(input("please enter cpu freq. (between 600 and 1800 MHz): "))

    while cpuFreq < 600 or cpuFreq > 1800:
        print("cpu freq must be between 600 and 1800 MHz")
        max_chunk_mult = int(input("please enter new cpu freq.: "))
    
    run(["sudo","cpufreq-set","--max","{}Mhz".format(cpuFreq)])
    print(cpu_freq())

    data_name = str(input("please enter dataset name: "))

    dataset_type = input("please enter dataset type (random vs pattern): ")

    if dataset_type == 'random':

        chunk_size = int(input("please enter chunk size (min. 60 sec.): "))
        
        while chunk_size < 60:
            print("minimum chunk size is 60 sec.")
            chunk_size = int(input("please enter chunk size: "))
        
        total_length = int(input("please enter total run time (min. {} sec.): ".format(chunk_size*4)))
        
        while total_length < chunk_size*4:
            print("minimum total run time is {} sec. ".format(chunk_size*4))
            total_length = int(input("please enter new total time: "))

        max_chunk_mult = int(input("please enter maximum chunk multiple: "))
        
        while max_chunk_mult < 0:
            print("multiple must be greater than 0")
            max_chunk_mult = int(input("please enter new maximum chunk multiple: "))

        switch_net = input("switch between WiFi and LTE? (y/n): ")

        if switch_net == 'Y':
            switch_net = 'y'

        return interval, total_length, chunk_size, max_chunk_mult, switch_net, cpuFreq, data_name, dataset_type

    if dataset_type == 'pattern':

        total_length = int(input("please enter total run time: "))

        while total_length < 0:
            print("multiple must be greater than 0")
            total_length = int(input("please enter new maximum chunk multiple: "))


        return interval, total_length, 0, 0, 'n', cpuFreq, data_name, dataset_type

    
    
    

        

def Connect(net):
    """Switches Network Connection"""
    print('Connected to {}'.format(net['name']))
    Popen(['sudo','raspi-config'])
    
    keyboard.press(Key.enter)
    keyboard.release(Key.enter)
    sleep(1.5)
    keyboard.press(Key.enter)
    keyboard.release(Key.enter)
    sleep(1.5)
    keyboard.press(Key.enter)
    keyboard.release(Key.enter)
    sleep(1.5)
    
    keyboard.type(net['ssid'])
    
    keyboard.press(Key.enter)
    keyboard.release(Key.enter)
    sleep(1.5)
    
    keyboard.type(net['psk'])
    
    keyboard.press(Key.enter)
    keyboard.release(Key.enter)
    sleep(1.5)
    
    keyboard.press(Key.esc)
    keyboard.release(Key.esc)
        
def augmented_reality(length = 60):
    """Runs Augmented Reality Application"""
    print("Running Augmented Reality for {} seconds".format(length))
    
    os.chdir('/home/pi/u-worc')
    
    # load the source image from disk
    source = cv2.imread("figures/source.png")
    # load the ArUCo dictionary, grab the ArUCo parameters, and detect the markers
    arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_ARUCO_ORIGINAL)
    arucoParams = cv2.aruco.DetectorParameters_create()

    t_end = time() + length
    print("AR ends at {}".format(strftime('%H:%M:%S', localtime(t_end))))

    while time() < t_end:
        
        image = cv2.imread("figures/aruco_input.jpg")
        image = imutils.resize(image, width=600)
        (imgH, imgW) = image.shape[:2]
        image = imutils.rotate(image, choice([-270, -180, -90, 90, 180, 270]))

        (corners, ids, rejected) = cv2.aruco.detectMarkers(image, arucoDict,
                                                        parameters=arucoParams)
        # if we have not found four markers in the input image then we cannot
        # apply our augmented reality technique
        if len(corners) != 4:
            print("[INFO] could not find 4 corners...exiting")
            sys.exit(0)
        # otherwise, we've found the four ArUco markers, so we can continue
        # by flattening the ArUco IDs list and initializing our list of
        # reference points
        ids = ids.flatten()
        refPts = []
        # loop over the IDs of the ArUco markers in top-left, top-right,
        # bottom-right, and bottom-left order
        for i in (923, 1001, 241, 1007):
            # grab the index of the corner with the current ID and append the
            # corner (x, y)-coordinates to our list of reference points
            j = np.squeeze(np.where(ids == i))
            corner = np.squeeze(corners[j])
            refPts.append(corner)

        # unpack our ArUco reference points and use the reference points to
        # define the *destination* transform matrix, making sure the points
        # are specified in top-left, top-right, bottom-right, and bottom-left
        # order
        (refPtTL, refPtTR, refPtBR, refPtBL) = refPts
        dstMat = [refPtTL[0], refPtTR[1], refPtBR[2], refPtBL[3]]
        dstMat = np.array(dstMat)
        # grab the spatial dimensions of the source image and define the
        # transform matrix for the *source* image in top-left, top-right,
        # bottom-right, and bottom-left order
        (srcH, srcW) = source.shape[:2]
        srcMat = np.array([[0, 0], [srcW, 0], [srcW, srcH], [0, srcH]])
        # compute the homography matrix and then warp the source image to the
        # destination based on the homography
        (H, _) = cv2.findHomography(srcMat, dstMat)
        warped = cv2.warpPerspective(source, H, (imgW, imgH))

        # construct a mask for the source image now that the perspective warp
        # has taken place (we'll need this mask to copy the source image into
        # the destination)
        mask = np.zeros((imgH, imgW), dtype="uint8")
        cv2.fillConvexPoly(mask, dstMat.astype("int32"), (255, 255, 255),
                        cv2.LINE_AA)
        # this step is optional, but to give the source image a black border
        # surrounding it when applied to the source image, you can apply a
        # dilation operation
        rect = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.dilate(mask, rect, iterations=2)
        # create a three channel version of the mask by stacking it depth-wise,
        # such that we can copy the warped source image into the input image
        maskScaled = mask.copy() / 255.0
        maskScaled = np.dstack([maskScaled] * 3)
        # copy the warped source image into the input image by (1) multiplying
        # the warped image and masked together, (2) multiplying the original
        # input image with the mask (giving more weight to the input where
        # there *ARE NOT* masked pixels), and (3) adding the resulting
        # multiplications together
        warpedMultiplied = cv2.multiply(warped.astype("float"), maskScaled)
        imageMultiplied = cv2.multiply(image.astype(float), 1.0 - maskScaled)
        output = cv2.add(warpedMultiplied, imageMultiplied)
        output = output.astype("uint8")

        # show the input image, source image, output of our augmented reality
        cv2.imshow("Input", image)
        #cv2.imshow("Source", source)
        cv2.imshow("OpenCV AR Output", output)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
      
    cv2.destroyAllWindows()         


        

def mine_DuinoCoin(length = 60):
    print("Running DuinoCoin miner for {} seconds".format(length))
    
    os.chdir('/home/pi/duino-coin')
    
    mine = Popen(["python","PC_Miner.py"], preexec_fn=os.setsid)
    
    sleep(1)
    keyboard.type('n')
    sleep(0.5)
    keyboard.press(Key.enter)
    keyboard.release(Key.enter)
    
    print("mining ends at {}".format(strftime('%H:%M:%S', localtime(time() + length))))

    sleep(length)

    os.killpg(os.getpgid(mine.pid), SIGTERM)

def _idiot_run():# run around like an idiot
    keyboard.press(Key.space)
    sleep(0.2)
    keyboard.release(Key.space)

    keyboard.press(Key.up)
    sleep(2.5)
    keyboard.release(Key.up)

    keyboard.press(Key.left)
    sleep(1.5)
    keyboard.release(Key.left)

def doom(length = 60):
    print("Running Doom for {} seconds".format(length))

    os.chdir("/usr/games")
    game = Popen(["chocolate-doom","-iwad",'DOOM1.WAD'])
    sleep(6)# wait for game to start

    keyboard.press(Key.space)
    keyboard.release(Key.space)

    # start game
    for _ in range(3):
        keyboard.press(Key.enter)
        keyboard.release(Key.enter)
        sleep(3)
    
   
    t_end = time() + length - 15
    print("game ends at {}".format(strftime('%H:%M:%S', localtime(t_end))))


    while time() < t_end: # move around in game
        _idiot_run()

    game.kill()

#to open youtube and play video
def stream(length = 60):
    print("Running Youtube for {} seconds".format(length))
    stream = Popen(["firefox-esr", "https://www.youtube.com/watch?v=T75IKSXVXlc"])
    print("streaming ends at {}".format(strftime('%H:%M:%S', localtime(time() + length - 15))))


    sleep(15)

    print("starting video")
    keyboard.press(Key.space)
    sleep(0.1)
    keyboard.release(Key.space)

    sleep(length - 15)

    stream.kill()

def do_nothing(length = 60): # sit idly
    print("Taking Break for {} seconds".format(length))

    # chill
    t_end = time() + length
    print("Ends at {}".format(strftime('%H:%M:%S', localtime(t_end))))

    sleep(length)


def get_lengths(total_length, chunk_size, max = 4):
    """Returns list of state time lengths
    @params:
        total_length: length of dataset
        chunk_size: length of minimum time spent in state
    """
    final_chunks = []

    if chunk_size != 0:
        
        chunks = int(total_length / chunk_size)
        chunks_list = [chunk_size for _ in range(chunks)]
        
        
        chunky, count = chunk_size, 0

        for index, chunk in enumerate(chunks_list):
        
            if bool(getrandbits(1)) == True and count < max:
                
                if index == len(chunks_list) - 1:
                    final_chunks.append(chunky)
                    
                else:
                    count += 1
                    chunky += chunk
            else:
                final_chunks.append(chunky)
                
                chunky, count = chunk_size, 0
    
    else:
        temp_chunks = [600]*4+[1200]*4+[1800]*4+[3600]*2
        shuffle(temp_chunks)
        curr_length, curr_indx = 0, 0

        while curr_length < total_length: # keep adding chunks based on random shuffle of temp chunks until total length passed

            curr_length += temp_chunks[curr_indx]
            final_chunks.append(temp_chunks[curr_indx])
            curr_indx += 1
            
            if curr_indx == 14:
                curr_indx = 0

        final_chunks[-1] = final_chunks[-1] - (curr_length - total_length) # remove excess chunk time from last chunk

    return final_chunks

def get_current():
    
    usage_data = monitor.get_current_usage()
    usage_data.to_csv("/home/pi/u-worc/data/{}_{}MHz_res_usage_data_{}.csv".format(worker_name, cpuFreq, data_name))
        
    print("saving monitoring data in '{}_{}MHz_res_usage_data_{}.csv'".format(worker_name, cpuFreq, data_name)) 

def run_prog(states, nets, chunks_list, switch_net, cpuFreq, data_name, dataset_type):
    
    past_net = None
    net_name = ''
    states_pattern = []
    state_pattern_indx = 0
    thresh = len(states) - 2
    
    for time_chunk in chunks_list:

        

        state = choice(states)
        net = choice(nets)
        
        states.remove(state)

        if past_net != net and switch_net == 'y':
            Connect(net)
            sleep(5)
            net_name = net['name']

        if len(states) <= thresh:
            states.append(past_state)

        past_state = state
        past_net = net
        
        if dataset_type == 'pattern':

            if len(states_pattern) < 14:
                
                # save pattern
                states_pattern.append(state)
            
            else:

                #loop over pattern
                
                state = states_pattern[state_pattern_indx]
                
                state_pattern_indx += 1

                
                if state_pattern_indx  == 14:
                    state_pattern_indx = 0

        
        monitor.state = state['name']  # Sets state as in monitoring data until state begins
        monitor.net = net_name  # Saves net with monitoring data

        try:
            
            state["func"](time_chunk)

            t1 = Thread(target = get_current) 
            t1.daemon = True
            t1.start()                
            
        except:

            print("Error saving monitoring data")     


        

    t1.join()
    print('stopping monitor and saving data')
    usage_data = monitor.stop_monitor_thread()
    usage_data.to_csv("/home/pi/u-worc/data/{}_{}MHz_res_usage_data_{}.csv".format(worker_name, cpuFreq, data_name))

    run(["sudo","cpufreq-set","--max","{}Mhz".format(1800)])
    print(cpu_freq())

if __name__ == "__main__":
    
    keyboard = Controller()
    worker_name = platform.node()
    worker_seeds = {'RPi400':42,'RPi4B8GB':101,'RPi4B4GB':666,'RPi4B2GB1':999,'RPi4B2GB2':1111}
    seed(worker_seeds[worker_name])

    states = [{"name":"augmented_reality","func":augmented_reality},{"name":"mining","func":mine_DuinoCoin},{"name":"game","func":doom}, {"name":"idle","func":do_nothing}, {"name":"stream","func":stream}]

    nets_1 = [{"name":"WiFi","ssid":"X","psk":"x"},{"name":"LTE","ssid":"Z","psk":"z"}]
    nets_2 = [{"name":"WiFi","ssid":"Y","psk":"y"},{"name":"LTE","ssid":"Z","psk":"z"}]

    #"""Input Configuration Args"""
    interval, total_length, chunk_size, max_chunk_mult, switch_net, cpuFreq, data_name, dataset_type = start()

    #"""Configure Monitoring Interval Start Resource Usage Monitor""""
    global monitor
    monitor = Usage(interval = interval)
    

    print("starting monitor")

    monitor.run_monitor_thread(ipykernel = False, topProc = False)
    
    #"""Generate RU State Schedule"""
    chunks_list = get_lengths(total_length, chunk_size, max_chunk_mult)

    #"""Run Dynamic Usage Prog."""
    run_prog(states, nets_1, chunks_list, switch_net, cpuFreq, data_name, dataset_type)

    






