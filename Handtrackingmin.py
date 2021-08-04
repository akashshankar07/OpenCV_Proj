import cv2
import mediapipe as mp
import time

from mediapipe.python.solutions import hands

cap= cv2.VideoCapture(0)

mpHands=mp.solutions.hands
hands=mpHands.Hands()
mpDraw=mp.solutions.drawing_utils

pTime=0
cTime=0

while True:
    success, img= cap.read()
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    res = hands.process(imgRGB)
    
    if res.multi_hand_landmarks:
        for handslm in res.multi_hand_landmarks:
            #ID and corresponding landmarks
            for id,lm in enumerate(handslm.landmark):
                #the landmarks give the ratio of the image, on multiplying with width and height we get pixel value
                print(id,lm)
                #height,width and channels of the img
                h ,w ,c = img.shape
                cx=int(lm.x*w)
                cy=int(lm.y*h)
                if id==0:
                    #0 is base, 4 is tip of the thumb
                    cv2.circle(img, (cx,cy),30,(64,0,255),cv2.FILLED)


            mpDraw.draw_landmarks(img,handslm,mpHands.HAND_CONNECTIONS)

    cTime=time.time()
    fps = 1/(cTime-pTime)
    pTime=cTime 

    cv2.putText(img,str(int(fps)),(10,75),cv2.FONT_HERSHEY_SIMPLEX,3,(250,0,250),3)



    cv2.imshow("image",img)
    cv2.waitKey(1)
    

    