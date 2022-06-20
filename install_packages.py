from subprocess import Popen, check_call 
import sys
import os
from time import sleep
#to install libraries automatically
import pip

def install(package):
    """Automatically installs python pip package and restarts the program"""
    print("automatically installing {} with pip".format(package))
    
    pip.main(["install",  package])


def misc_install(package):
    """Automatically installs apt package and restarts the program"""
    print("automatically installing {}".format(package))

    if package == "duino-coin":
        os.chdir("/home/pi")
        
        clone = Popen(['git','clone', "https://github.com/revoxhere/duino-coin"])
        try:
            clone.wait(timeout=120)
            print("cloning done")
        except:
            print("time's up, duino-coin not cloned")
            
        os.chdir("duino-coin")
        
        print("please configure miner (3-min timer set)")
        mine = Popen(["python","PC_Miner.py"], preexec_fn=os.setsid)

        sleep(180)

        os.killpg(os.getpgid(mine.pid), SIGTERM)
        os.chdir('/home/pi')
        
    elif package == "firefox-esr":
        ffesr = Popen(['sudo','apt','install', package])
        try:
            ffesr.wait(timeout=300)
            print("install done")
        except:
            print("time's up, browser not installed")

    elif  package == "chocolate-doom":
        os.chdir("/usr/games")
        dmdwnld = Popen(['sudo','wget', "http://www.doomworld.com/3ddownloads/ports/shareware_doom_iwad.zip"])
        try:
            dmdwnld.wait(timeout=120)
            print("download done")
        except:
            print("time's up, doom not downloaded")
        
        Popen(['sudo','unzip', "shareware_doom_iwad.zip"])
        
        setup = Popen(['chocolate-doom-setup']) #change control and remove fullscreen
        
        try:
            setup.wait(timeout=300)
            print("setup done")
        except:
            print("time's up, doom not setup")
        
        os.chdir('/home/pi')