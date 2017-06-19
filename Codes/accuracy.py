import os
import json
import pickle
import numpy as np

'''
Classification Accuracy
'''

groundtruth_label = np.load('./../Standalone/testing_label.npy')
predicted_label = np.load('./../Standalone/predicted_label.npy')

count_vehicle = 0
count_pedestrian = 0
total_vehicle = 0
total_pedestrian = 0

for i in range(len(groundtruth_label)):
	if int(groundtruth_label[i])==0:
		if groundtruth_label[i]==predicted_label[i]:
			count_pedestrian+=1
		total_pedestrian+=1
	elif int(groundtruth_label[i])==1:
		if groundtruth_label[i]==predicted_label[i]:
			count_vehicle+=1
		total_vehicle+=1

print "Accuracy of Classification for Vehicle: ", float(count_vehicle*100.0)/total_vehicle
print "Accuracy of Classification for Pedestrian: ", float(count_pedestrian*100.0)/total_pedestrian
print "Overall Classification Accuracy for QP=27 is: ", float((count_pedestrian+count_vehicle)*100.0)/len(predicted_label)
