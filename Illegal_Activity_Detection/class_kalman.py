from numpy import *


class Kalman(object):
	"""The standard implementtation of the object tracking using Kalman filters"""
	
	def __init__(self):

		"""Initialising constants required for calculations"""
		self.F = matrix([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]],dtype = 'float')
		self.H = matrix([[1,0,0,0],[0,1,0,0]],dtype = 'float')
		self.R = matrix(ones((2),dtype = 'float'))
		self.I =  matrix(eye((4),dtype = 'float'))
		self.err_z = matrix([[1000,0],[0,1000]],dtype = 'float')
		self.err_x = matrix([[1/4,0,1/2,0],[0,1/4,0,1/2],[1/2,0,1,0],[0,1/2,0,1]],dtype = 'float')
	
	
	def predict(self,X,P):
		
		"""The predict step for Kalman filtering
		Inputs
		
		Parameter 1 : numpy matrix of previous position and veloctiy
		
		Parameter 2 : numpy matrix, error of prediction

		Returns : Updated value of X and P
		"""

		X = (self.F*X)
		P = self.F*P*self.F.T + self.err_x
		
		return X,P

	def update(self,X,P,pos):

		"""The predict step for Kalman filtering
		Inputs
		
		Parameter 1 : numpy matrix of previous position and veloctiy
		
		Parameter 2 : numpy matrix, erro in prediction
		
		Parameter 3 : numpy matrix, measured position for new time step
		Returns : Updated value of X and P
		"""

		Z = matrix([[pos[0]],[pos[1]]])
		y = Z - self.H*X
		S = (self.H * P * self.H.T) + self.err_z
		K = P*self.H.T*S.I
		X = X + K*y
		P = (self.I - (K*self.H)) *(P)
	
	
		return X,P