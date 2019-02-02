import cv2
from numpy import *
import matplotlib.pyplot as plt

cap = cv2.VideoCapture('test.avi')
fgbg = cv2.createBackgroundSubtractorKNN()
X = matrix([[0],[0],[0],[0]],dtype = 'float')
F = matrix([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]],dtype = 'float')
H = matrix([[1,0,0,0],[0,1,0,0]],dtype = 'float')
R = matrix(ones((2),dtype = 'float'))
I =  matrix(eye((4),dtype = 'float'))

err_z = matrix([[1000,0],[0,1000]],dtype = 'float')
err_x = matrix([[1/4,0,1/2,0],[0,1/4,0,1/2],[1/2,0,1,0],[0,1/2,0,1]],dtype = 'float')
P = err_x


while(cap.isOpened()):
	ret,frame = cap.read()
	if ret == False:
		break
	else:
		fgmask = fgbg.apply(frame)
		gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
		cv2.imshow('frame',fgmask)
		image,contours,hierarchy = cv2.findContours(fgmask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
		for i in contours:
			x,y,w,h = cv2.boundingRect(i)
			
		 	if w > 50 or h >50:
			 	
			 	centre = (x+(w/2),y+(h/2))
			 	cv2.circle(frame,centre,2,(0,255,2),2)
			 	m,n,p = frame.shape
				Z = matrix([[centre[0]],[centre[1]]])#matrix([[measurement[n1]]])
				
				y = Z - H*X
 				S = (H * P * H.T) + err_z
				K = P*H.T*S.I
				X = X + K*y
				P = (I - (K*H)) *(P)
				X = (F*X)
				P = F*P*F.T + err_x
				centre1 = (int(X[0]),int(X[1]))
				radius1 = int(2)
				cv2.circle(frame,centre1,radius1,(0,0,255),2)
				pos = str(centre1)#(int(X[0],int(X[1]))

				vel = int(sqrt(int(X[2])**2+(int(X[3])**2)))
				font = cv2.FONT_HERSHEY_SIMPLEX
				frame = cv2.putText(frame,'%s'%pos,(0,20), font, 0.7,(0,0,255),2,cv2.LINE_AA)
				frame = cv2.putText(frame,'%sm/sec'%vel,(0,50), font, 0.7,(0,0,255),2,cv2.LINE_AA)
	

				cv2.imshow('image',frame)
			else:
				cv2.imshow('image',frame)
	if cv2.waitKey(30) == ord('q'):
		break
cv2.destroyAllWindows()
cap.release()