import os
import cv2
import numpy as np

'''
input parameters
'''
#-----------------------------------------------

video_type = 'Pedestrian_dataset' # Vehicle_dataset, Pedestrian_dataset
video_name = 'hall_monitor'

#-----------------------------------------------

for frame in range(27,60):

	print frame
	img_1 = cv2.imread('./../'+video_type+'/'+video_name+'/photos/'+'final'+str(frame)+'.jpg',cv2.IMREAD_COLOR)
	img_2 = cv2.imread('./../'+video_type+'/'+video_name+'/photos/'+'newfinal'+str(frame)+'.jpg',cv2.IMREAD_COLOR)
	cv2.imshow('Original image',img_1)
	cv2.moveWindow("Original image", 20,20)
	cv2.imshow('New image',img_2)
	cv2.moveWindow("New image", 500,20)
	cv2.waitKey(0)
	cv2.destroyAllWindows()