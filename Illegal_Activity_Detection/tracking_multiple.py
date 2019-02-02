import cv2
from numpy import *
from class_kalman import Kalman
 
def find_index(final_X_temp,X_temp):
	
	index = 0 
	for val in X_temp:
		var = val - final_X_temp
		if  (var[0] + var[1] + var[2] + var[3] == 0):
			break
		index+=1
	
	return index


def find_least_distance(X_temp,X_predict):
	
	dist = zeros((len(X_temp)),dtype= 'float')
	
	for i in range(len(X_temp)):
		var = X_temp[i]
		dist[i] = linalg.norm(var[0:2] - X_predict[0:2])
	min_dist = amin(dist)
	dist = dist.tolist()
	index = dist.index(min_dist)
	final_X_temp = X_temp[index]

	return final_X_temp


def find_centre_indices(centres,predicts):
	indices = []
	m,n = len(centres),len(predicts)
	dist = zeros((m,n),dtype = 'float') 

	for i in range(m):
		centre = centres[i]
		centre = array(centre)
		for j in range(n):
			predict = predicts[j]
			predict = predict[0:2].T
			dist[i][j] = linalg.norm(predict - centre)
	
	
	dist =  dist.min(axis=1).tolist()
	for t in range(m-n):
		num = max(dist)
		index1 = dist.index(num)
	
		indices.append(centres[index1])
		dist[index1] = 0.0
	
	return indices

def centre_m_equ_n(centres,predicts):
	i,j = 0,0
	a = zeros((2,2),dtype = 'float')
	for centre in centres:
		for predict in predicts:
			dist = linalg.norm(matrix(centre) - predict[0:2].T)
			a[i][j] = dist 
		j+=1
	
	j=0	
	i+=1
	c = a.min(axis=1)
	mean_val = c.mean()
	thresh = mean_val + 20



def create_predict_list(X_update_list,P_update_list):
	X_predict_list,P_predict_list = [],[]
	n = len(X_update_list) 
	
	for j in range(0,n):
		X,P = X_update_list[j],P_update_list[j]
		X_predict,P_predict = kal.predict(X,P)
 		X_predict_list.append(X_predict)
 		P_predict_list.append(P_predict)
	
	return X_predict_list,P_predict_list


def create_update_list(centre_list,X_predict_list,P_predict_list,frame):
	global X_predict_initial,P_predict_initial,colors
	m,n = len(centre_list),len(X_predict_list)
	X_update_list,P_update_list = [],[]


	if (m > n):
		if X_predict_list == []:
			for centre_pt in centre_list:
				X_predict_list.append(matrix([[centre_pt[0]],[centre_pt[1]],[0],[0]]))
				P_predict_list.append(P_predict_initial)
		else:
			centre_indices = find_centre_indices(centre_list,X_predict_list)

			for brij in centre_indices:
				X_predict_list.append(matrix([[brij[0]],[brij[1]],[0],[0]]))
				P_predict_list.append(P_predict_initial)

	for j in range(0,len(X_predict_list)):
		X_temp,P_temp = [],[]
		X_predict,P_predict = X_predict_list[j],P_predict_list[j]
		for i in range(0,len(centre_list)):
			centre_val = centre_list[i]
			X_update,P_update = kal.update(X_predict,P_predict,centre_val)
			X_temp.append(X_update),P_temp.append(P_update)
		#Now I have three values of update the final update is the value closest to the predict value. This can be foun out by the f
		#find distance method
		
		final_X_temp = find_least_distance(X_temp,X_predict)
		
		
		
		index = find_index(final_X_temp,X_temp)
		final_P_temp = P_temp[index]
		X_update_list.append(final_X_temp)
		P_update_list.append(final_P_temp)
		if (len(X_update_list) == len(centre_list)):
			break
	
	# for k in range(len(X_update_list)):
	# 	X_temp_val = X_update_list[k]
	# 	centre2 = (int(X_temp_val[0]),int(X_temp_val[1]))
	# 	cv2.circle(frame,centre2,2,colors[k],2)



	
	return X_update_list,P_update_list



#for the frame of the video loop to find contours and centre  the loop over centre to find the predict values for every centre. For next frame 
#the centres there and find the update values for the centres and fron the predict information assign each object previous update values.Now
cap = cv2.VideoCapture('/home/brij/Desktop/new_dataset/harsh_cam/new_mar_7/use_case/2.MOV')
tofile = []
fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows = False)
fgbg.setDetectShadows(detectShadows = False)

X_predict_list = []#matrix([[0],[0],[0],[0]],dtype = 'float')
P_predict_initial = matrix([[1/4,0,1/2,0],[0,1/4,0,1/2],[1/2,0,1,0],[0,1/2,0,1]],dtype = 'float')
P_predict_list = []
kal = Kalman()
kernel = ones((2,2),uint8)
frame_cnt = 0
colors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255),(255,255,255),(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255),(255,255,255)]
while(cap.isOpened()):
	frame_cnt+=1
	centre_list = []
	ret,frame = cap.read()
	if ret == True:
		fgmask = fgbg.apply(frame)
		fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
		gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
		cv2.imshow('frame',fgmask)
		image,contours,hierarchy = cv2.findContours(fgmask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

		for cnt in contours:
			x,y,w,h = cv2.boundingRect(cnt)
		 	if w>50 or h>50:	
		 		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,233,122),2)
		 		centre = [x+(w/2),y+(h/2)]
		 		centre_list.append(centre)
			
	 	if (len(centre_list) == 0):
	 		continue
	 	cv2.imshow('frame123',frame)

		if frame_cnt < 6:
			continue
		#Suppose I have three elements in predict_mat of first iteration 
		#This is the second iteration I ll have three centre values and using three centre and three predcit ill calculate
		#9 update Ie one update value out of three for ine obect will be closest to one of the predict values likewise Ill
		#assign it to the object usong the hungarian algorithm

		X_update_list,P_update_list = create_update_list(centre_list,X_predict_list,P_predict_list,frame)

		tofile.append(X_update_list[0][0:2])
		for k in range(len(X_update_list)):
			X_temp_val = X_update_list[k]
			centre2 = (int(X_temp_val[0]),int(X_temp_val[1]))
			cv2.circle(frame,centre2,2,colors[k],2)
		cv2.ellipse(frame,(417, 360),(283, 228),0,0,360,(255,0,0),2)

		X_predict_list,P_predict_list = create_predict_list(X_update_list,P_update_list)
		cv2.imshow('finalframe',frame)
	
	else:
		break
	if cv2.waitKey(0) == ord('q'):
		continue
	if cv2.waitKey(30) == ord('w'):
		break

cv2.destroyAllWindows()
cap.release()