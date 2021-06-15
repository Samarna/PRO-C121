import numpy as np
import cv2 
import time

fourcc = cv2.VideoWriter_fourcc(*'XVID')
# 20.0 is frames per second
# (640,480) is frame size
output_file = cv2.VideoWriter('output.avi',fourcc,20.0,(640,480))
# starting camera
cam = cv2.VideoCapture(0)
time.sleep(2)
bg = 0
for i in range(60):
    ret,bg = cam.read()
bg = np.flip(bg,axis=1)
while(cam.isOpened()):
    ret,img = cam.read()
    if not ret:
        break
    img = np.flip(img,axis=1)
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    l_black = np.array([104,153,70])
    u_black = np.array([30,30,0])
    mask_1 = cv2.inRange(hsv,l_black,u_black)

    #l_black = np.array([170,120,70])
    #u_black = np.array([180,255,255])
    mask_2 = cv2.inRange(hsv,l_black,u_black)
    final_mask = mask_1 + mask_2
    final_mask = cv2.morphologyEx(final_mask,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    final_mask = cv2.morphologyEx(final_mask,cv2.MORPH_DILATE,np.ones((3,3),np.uint8))
    mask_2 = cv2.bitwise_not(final_mask)

    result_1 = cv2.bitwise_and(img,img,mask=mask_2)
    result_2 = cv2.bitwise_and(bg,bg,mask=final_mask)
    final_output = cv2.addWeighted(result_1,1,result_2,1,0)
    output_file.write(final_output)
    cv2.imshow("MAGIC",final_output)
    cv2.waitKey(1)

cam.release()
cv2.destroyAllWindows()
