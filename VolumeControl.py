import cv2
import time
import numpy as np
import HandTrackModule as htm
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

#setting width and height of cam
wCam,hCam = 1280, 720

capt = cv2.VideoCapture(0)
capt.set(3,wCam)
capt.set(4,hCam)
pTime=0
cTime=0
vol=0
volper=0
volbar=400

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

#volume.GetMute()
#volume.GetMasterVolumeLevel()
rangevol = volume.GetVolumeRange()
#volume.SetMasterVolumeLevel(-20.0, None)
minvol = rangevol[0]
maxvol = rangevol[1]


detector = htm.handDetector(detectionConf=0.7)

while True:
    success, img = capt.read()
    img = detector.findHands(img)
    landmarklst = detector.findPos(img,draw=False)
    if len(landmarklst) !=0 :
        #print(landmarklst[4],landmarklst[8])
    #for thumb and index finger value number 4 and 8. Base is 0
        a1,b1=landmarklst[4][1],landmarklst[4][2]
        a2,b2=landmarklst[8][1],landmarklst[8][2]
        #center co-ord
        cx,cy=(a1+b1)//2,(a2+b2)//2
        
        
        cv2.circle(img,(a1,b1),15,(64,0,255),cv2.FILLED)
        cv2.circle(img,(a2,b2),15,(64,0,255),cv2.FILLED) 
    #drawing a line in between
        cv2.line(img, (a1,b1),(a2,b2),(255,0,255),3)
        cv2.circle(img,(cx,cy),15,(64,0,255),cv2.FILLED)

        leng = math.hypot(a2-a1,b2-b1)
        #minimum comes around 50

        #conversion of hand range from length to equivalent volume range
        vol =np.interp(leng, [50,300],[minvol,maxvol])
        volbar=np.interp(leng, [50,250],[400,150])
        volper=np.interp(leng,[50,220],[0,100])
        print(vol)
        volume.SetMasterVolumeLevel(vol, None)


        if leng<50:
            cv2.circle(img,(cx,cy),15,(64,0,255),cv2.FILLED)
            #different color when fingers are closed

    #rectangle bar for volume
    cv2.rectangle(img,(55,150),(84,400),(250,0,250),4)
    cv2.rectangle(img,(55,int(volbar)),(84,400),(0,255,0),cv2.FILLED)
    cv2.putText(img,f'{int(volper)} %',(40,460),cv2.FONT_HERSHEY_SIMPLEX,3,(250,0,250),3)


    cTime=time.time()
    fps = 1/(cTime-pTime)
    pTime=cTime
    cv2.putText(img,str(int(fps)),(10,75),cv2.FONT_HERSHEY_SIMPLEX,3,(250,0,250),3)
       
    cv2.imshow("img",img)
    cv2.waitKey(1)