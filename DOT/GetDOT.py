import pandas as pd
import pyautogui as gui
import keyboard
import time
time.sleep(2)
for i in range(0,100):
    gui.moveTo(1030, 500, duration = .1) #Drop Down
    gui.click(duration=.1) #x,y
    
    gui.moveTo(1051, 900, duration = .5) #expand list
    gui.dragTo(1300, 900, duration = .2, button = "left")
    
    gui.moveTo(1230, 980, duration = .2) #Click through list
    gui.click(duration = .1) #x,y
    gui.moveTo(750, 550, duration = .1) #Select first on list
#gui.moveTo(1030, 500, duration = .1) #Close Drop Down
    gui.click(duration=.5) #x,y
    gui.moveTo(300, 350, duration = .1) #Country name
    time.sleep(5) #wait for it to load
    gui.click(duration=.1)
    gui.moveTo(300, 450, duration = .1)
    gui.click(duration=.5)
    gui.moveTo(700, 1250, duration = .1)
    gui.click(duration=.1)
    gui.moveTo(1160, 150, duration = .1) #Download
    time.sleep(10)
    
    #gui.click(duration=.5)
    
    #gui.click()
    #gui.click()

    #gui.keyDown("Ctrl")
    #gui.press("C")
    #gui.keyUp("Ctrl")
    

#gui.typewrite("#Hello World", interval = .2)

#gui.keyDown("shift")

#gui.write('uppercase')

#gui.keyUp("shift")#Hello WorldUPPERCASE

#gui.keyDown("Ctrl")

#gui.keyUp("Ctrl")