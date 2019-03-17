#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2

cam = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_alt2.xml')

i = 0
while True:
    ret , capture = cam.read()

    gray = cv2.cvtColor(capture , cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray , scaleFactor= 1.5, minNeighbors = 5)

    for (x, y , w, h) in faces:
        roi_color = capture[y-50:y+h+50, x-50:x+w+50]
        break
        
    #colecting the datafor the facial recognition
    name = 'images/fairandlovely/Images_data_{}.jpg'.format(i)
    i = i+1
    cv2.imwrite(name ,  roi_color)
        
    cv2.imshow('FACIALRECOGNITION', roi_color)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()


# In[ ]:




