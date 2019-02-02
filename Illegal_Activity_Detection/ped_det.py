import cv2
from numpy import *
from sklearn.externals import joblib
from skimage.feature import hog
from sklearn.svm import LinearSVC
import os
import matplotlib.pyplot as plt
import nms


def hog_det(frame):
	boundingboxes = []
	gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

	#Assuming we have a trained SVM model stroed in pickle file
	clf = joblib.load("/home/brij/python_opencv/real/pedestrians_det.pkl")
	win_size = (64, 128)
	m,n = gray.shape
	i,j,i1,j1=128,64,0,0
	iter_=0 
	while (i<=m):
		while(j<=n):
		 	mask = gray[i1:i,j1:j]

			hog0 = hog(mask, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2), visualise=False)
			temp = clf.predict(hog0)
			print temp
			if temp == [1]:

				points = j1,i1,j1+64,i1+128
				boundingboxes.append(points)

			j=j+8
			j1 = j1+8
		j = 64
		j1=0
		i = i+8
		i1= i1+8
	boundingboxes = asarray(boundingboxes)
	pick = nms.non_max_suppression_slow(boundingboxes, 0.3)
	for (startX, startY, endX, endY) in pick:
		cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
	
	
	return frame

if __name__ == '__main__':
	image = cv2.imread('/home/brij/Desktop/test_images/2.jpg')

	var= hog_det(image)
	cv2.imshow('image1',var)

	cv2.waitKey(0)
	cv2.destroyAllWindows() 




	cap = cv2.VideoCapture('test.avi')
	while(cap.isOpened()):
		ret,frame = cap.read()
		if ret == False:
			break
		else:
			var  = hog_det(frame)
		

		cv2.imshow('image',var)
		if cv2.waitKey(1) == ord('q'):
			break
	cv2.destroyAllWindows()
	cap.release()