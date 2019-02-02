import cv2
import time
from math import sqrt
from collections import deque
from numpy import * 
from matplotlib import pyplot as plt
from matplotlib.mlab import normpdf,bivariate_normal
from mpl_toolkits.mplot3d import Axes3D
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

global vals_data,mean_list,variance,vals_list,cir_buffer,covar

def plot(vals,mean_array):
	plt.figure(1)
	plt.subplot(211)
	plt.scatter(vals[:,0],vals[:,1],10,'r','o')
	plt.subplot(212)
	plt.plot (mean_array[:,0],mean_array[:,1],20,'r','^')
	plt.show()
	return

def val_at_variance(mean,covar,variance):
	const1 = matrix((variance - mean))
	const2 = const1*linalg.inv(covar)*const1.T
	exp_term = ((-1/2)*const2)
	probab = exp(exp_term)
	print "probab is ", probab
	return probab

def before_converge():

	global count
	plt.ion()
	diff = 100
	for val in vals_data:
		vals_list.append(val)
		vals = array(vals_list)
		mean = vals.mean(axis = 0)


		if count == 0:
			mean_temp = mean	
		else :
			diff = abs(mean_temp[0] - mean[0])
			if diff < 0.1 and count >100:
				break

		mean_temp = mean
		mean_list.append(mean)
		mean_array = array(mean_list)
		plt.figure(1)
		plt.subplot(211)
		plt.scatter(vals[:,0],vals[:,1],10,'r','o')
		plt.subplot(212)
		plt.scatter(mean_array[:,0],mean_array[:,1],80,'b','^')
		plt.draw()
		variance = vals.std(axis = 0)
		covarxy = vals.std()
		m,n = vals.shape
		covar = cov(vals[:,0],vals[:,1])

		count+=1

	cir_buffer = deque(vals_list,maxlen = count)
	print len(cir_buffer)
	return mean,variance,covar,cir_buffer

def after_converge(mean,covar,cir_buffer,threshold):
	cnt = 0
	mean_list1,illegal = [],[]
	plt.figure(1)
	plt.ion()
	plt.show()
	for i in range(count,len(vals_data)):
		#Calculate the gaussian value and if the value is greater than some threshold declare it as legal use it to update the gaussian 
		#and if it is less than threshold declare it as illegal
		plt.clf()
			#start prediction
		val_func = vals_data[i]
		const = 1/(linalg.det(covar))
		const1 = matrix((val_func - mean))
		const2 = const1*linalg.inv(covar)*const1.T
		exp_term = ((-1/2)*const2)
		probab = exp(exp_term)
		#print probab
		if probab >  threshold:

			if count >= 824:

				pass
			cir_buffer.append(val_func)
			vals_func = array(cir_buffer)
			mean = vals_func.mean(axis = 0)
			mean_list.append(mean)
			mean_array = array(mean_list)
			
			#plt.subplot(211)
			X,Y = meshgrid(vals_func[:,0],vals_func[:,1])
			z = bivariate_normal(X,Y,covar[0,0],covar[1,1],mean[0],mean[1],covar[0,1])
			plt.contour(X,Y,z)
			plt.scatter(vals_func[:,0],vals_func[:,1],10,'b','o')
			plt.scatter(mean[0],mean[1],100,'b','^')
			if illegal == []:
				pass
			else:
				plt.scatter(illegal_array[:,0],illegal_array[:,1],10,'r','o')

			time.sleep(2)
			plt.draw()
		else :
		
			if count >= 824:
				
				pass
			cnt+=1
			illegal.append(val_func)
			illegal_array = array(illegal)
	
		count += 1
	print cnt 
		
cap = cv2.VideoCapture('/home/brij/Desktop/new_dataset/harsh_cam/new_mar_7/use_case/2.MOV')
ret,frame = cap.read()

vals_data = loadtxt('/home/brij/python_work/opencv/real/files/new_data2_orig.txt')
vals_list,mean_list = [],[]
count = 0

mean,variance,covar,cir_buffer = before_converge()
mean1,variance1 = (int(mean[0]),int(mean[1])),(int(variance[0]),int(variance[1]))


threshold = val_at_variance(mean,covar,variance)
after_converge(mean,covar,cir_buffer,threshold)
plt.show()