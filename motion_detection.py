import cv2
import numpy as np

cap = cv2.VideoCapture(0)
background_image = None
while True:

    ret,frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(21,21),0)
    if background_image is None:
        background_image = gray
        continue
    delta_image = cv2.absdiff(gray,background_image)
    thresh = cv2.threshold(delta_image,18,255,cv2.THRESH_BINARY)[1]
    cnts = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[1]
    for cnt in cnts:
        if cv2.contourArea(cnt) < 10000:
            continue
        area = cv2.contourArea(cnt)
        (x,y,w,h) = cv2.boundingRect(cnt)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
        cv2.putText(frame,str(area), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
    cv2.imshow('delta_image',delta_image)
    cv2.imshow('frame',frame)
    cv2.imshow('thresh',thresh)
    k = cv2.waitKey(1)
    if k == 1 :
        break
cap.release()
cv2.destroyAllWindows()
