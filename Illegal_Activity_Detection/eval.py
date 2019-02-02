import cv2
from numpy import *

eval0 = loadtxt('/home/brij/eval.txt')
eval1 = loadtxt('/home/brij/eval1.txt')
confusion_mat = zeros((2,2),dtype = 'float')

for i in range(len(eval0)):
	var1 = eval0[i]
	var2 = eval1[i]

	if var1 == 1 and var2 == 1:
		confusion_mat[0][0] += 1

	elif var1 == 0 and var2 == 1 :
		confusion_mat[0][1] += 1

	elif var1 == 1 and var2 == 0:
		confusion_mat[1][0] += 1

	else :
		confusion_mat[1][1] += 1

true_pos,true_neg,false_pos,false_neg = confusion_mat[0][0], confusion_mat[1][1],confusion_mat[0][1],confusion_mat[1][0] 

precision = (true_pos/(true_pos+false_pos)) * 100
recall = (true_pos/(true_pos+false_neg)) *100
accuracy = ((true_pos + true_neg)/(true_pos + true_neg + false_pos +false_neg)) * 100
print confusion_mat
print precision,recall,accuracy