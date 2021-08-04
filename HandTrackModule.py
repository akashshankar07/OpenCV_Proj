import cv2
import mediapipe as mp
import time
import math

from mediapipe.python.solutions import hands

class handDetector():
    def __init__(self,mode=False,maxHands=2,detectionConf=0.5,trackConf=0.5):
        self.mode=mode
        self.maxHands = maxHands
        self.detectionConf= detectionConf
        self.trackConf= trackConf

        self.mpHands=mp.solutions.hands
        self.hands=self.mpHands.Hands(self.mode,self.maxHands,self.detectionConf,self.trackConf)
        self.mpDraw=mp.solutions.drawing_utils
        self.tipIDs = [4,8,12,16,20]
        #tip points of all the fingers

    def findHands(self,img,draw = True):

        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.res = self.hands.process(imgRGB)
            
        if self.res.multi_hand_landmarks:
                for handslm in self.res.multi_hand_landmarks:
                    if draw:
                        self.mpDraw.draw_landmarks(img,handslm,self.mpHands.HAND_CONNECTIONS)
        return img

    #findpos method to find the position of the image and parameter can have image attributes
    def findPos(self,img, handNo=0, draw=True):
        #creating a list of the landmarks
        xList=[]
        yList=[]
        boxx=[]
        self.lmLists=[]
        if self.res.multi_hand_landmarks:
            myHand = self.res.multi_hand_landmarks[handNo]
            #ID and corresponding landmarks
            for id,lm in enumerate(myHand.landmark):
                    #the landmarks give the ratio of the image, on multiplying with width and height we get pixel value
                    #print(id,lm)
                    #height,width and channels of the img
                    h ,w ,c = img.shape
                    cx=int(lm.x*w)
                    cy=int(lm.y*h)
                    xList.append(cx)
                    yList.append(cy)

                    self.lmLists.append([id,cx,cy])
                    if draw:
                        #0 is base, 4 is tip of the thumb
                        cv2.circle(img, (cx,cy),8,(64,0,255),cv2.FILLED)

            xmin,xmax = min(xList), max(xList)
            ymin,ymax = min(yList), max(yList)
            boxx=xmin,ymin,xmax,ymax

            if draw:
                cv2.rectangle(img,(xmin-15,ymin-15),(xmax+15,ymax+15),(0,255,0),2)
        return self.lmLists,boxx

    def fingersUP(self):
        fingers=[]
        #for thumb it is
        if self.lmLists[self.tipIDs[0]][1] > self.lmLists[self.tipIDs[0] -1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        #other fingers
        for id in range(1,5):
            if self.lmLists[self.tipIDs[id]][2] < self.lmLists[self.tipIDs[id] -2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        #for total no of fingers -> fingers.count(1)        
        return fingers

    def distance(self,a1,a2,img,draw=True,r=15,t=4):
        x1,y1 = self.lmLists[a1][1:]
        x2,y2 = self.lmLists[a2][1:]
        cx = (x1+x2) // 2
        cy = (y1+y2) // 2

        if draw:
            cv2.line(img,(x1,y1),(x2,y2),(0,255,0),t)
            cv2.circle(img,(x1,y1),r,(255,0,255),cv2.FILLED)
            cv2.circle(img,(x2,y2),r,(255,0,255),cv2.FILLED)
            cv2.circle(img,(cx,cy),r,(255,0,255),cv2.FILLED)
            length = math.hypot(x2-x1, y2-y1)

        return length,img, [x1,y1,x2,y2,cx,cy]
                    
"""
cTime=time.time()
fps = 1/(cTime-pTime)
pTime=cTime 

cv2.putText(img,str(int(fps)),(10,75),cv2.FONT_HERSHEY_SIMPLEX,3,(250,0,250),3)



cv2.imshow("image",img)
cv2.waitKey(1)"""

def main():
    pTime=0
    cTime=0
    cap= cv2.VideoCapture(0)
    detector = handDetector()
    #default parameters already set
    while True:
        success, img= cap.read()
        img = detector.findHands(img)
        #lmLists=detector.findPos(img,draw=False)
        lmLists=detector.findPos(img)
        boxx=detector.findPos(img)
        if len(lmLists)!=0:
            print(lmLists[0]) #can change index value, prints values of particular index val
        cTime=time.time()
        fps = 1/(cTime-pTime)
        pTime=cTime 

        cv2.putText(img,str(int(fps)),(10,75),cv2.FONT_HERSHEY_SIMPLEX,3,(250,0,250),3)



        cv2.imshow("image",img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()
