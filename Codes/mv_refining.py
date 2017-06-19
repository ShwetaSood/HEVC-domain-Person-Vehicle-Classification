import os
import cv2
import sys
import math
import json
import time
import numpy as np
from ast import literal_eval

'''
input parameters
'''
#-----------------------------------------------

video_type = 'Pedestrian_dataset' # Vehicle_dataset, Pedestrian_dataset
video_name = 'data_0053'

#-----------------------------------------------

with open('./../Config/video_config.json') as data_file:
	video_data = json.load(data_file)

filename = './../'+video_type+'/'+video_name+'/decoder_files'
files = ['decoder_cupu.txt','decoder_pred.txt','decoder_mv.txt']
input_image_path = './../'+video_type+'/'+video_name+'/input'
input_images =  np.sort(os.listdir(input_image_path))

video_width = video_data[video_type][video_name]["width"]
video_height = video_data[video_type][video_name]["height"]
no_frames=  video_data[video_type][video_name]["no_frames"]

no_ctu_width = int(math.ceil(video_width/64.0))
no_ctu_height = int(math.ceil(video_height/64.0))

copy_spatial_mv_data = np.load('./../'+video_type+'/'+video_name+'/output/'+'pic_mv_filtering.npy')
spatial_mv_data = np.load('./../'+video_type+'/'+video_name+'/output/'+'pic_mv_filtering.npy')
org_spatial_mv_data = np.load('./../'+video_type+'/'+video_name+'/output/'+'pic_mv.npy')

[frames ,height, width] = spatial_mv_data.shape

def find_magnitude(mv_array):
	if len(mv_array)==1: #Intra coded => No MV
		return 0
	elif len(mv_array)==4 or len(mv_array)==2:
		return math.sqrt(math.pow(mv_array[-1],2)+math.pow(mv_array[-2],2))

start_frame_no = 2 # First frame is I frame
start_time = time.time()
for frame in range(start_frame_no, no_frames-1): #3 to 1698
	print frame
	for i in range(len(org_spatial_mv_data[frame])):
		for j in range(len(org_spatial_mv_data[frame,0,:])):
			left = 1
			right = 1
			bottom = 1
			top = 1

			if i==0:
				top = 0
			if j==0:
				left = 0
			if i==(len(org_spatial_mv_data[frame])-1):
				bottom = 0
			if j==(len(org_spatial_mv_data[frame,0,:])-1):
				right = 0
			if left:
				left_pu = literal_eval(org_spatial_mv_data[frame,i,j-1])
			else:
				left_pu = [0]

			if right:
				right_pu = literal_eval(org_spatial_mv_data[frame,i,j+1])
			else:
				right_pu = [0]
			if top:
				top_pu = literal_eval(org_spatial_mv_data[frame,i-1,j])
			else:
				top_pu = [0]
			if bottom:
				bottom_pu = literal_eval(org_spatial_mv_data[frame,i+1,j])
			else:
				bottom_pu = [0]

			non_zero_mv = 0

			if left_pu!=[0]:
				mag = float(find_magnitude(left_pu))
				if mag!=0.0:
					non_zero_mv+=1

			if right_pu!=[0]:
				mag = float(find_magnitude(right_pu))
				if mag!=0.0:
					non_zero_mv+=1
			if top_pu!=[0]:
				mag = float(find_magnitude(top_pu))
				if mag!=0.0:
					non_zero_mv+=1
			if bottom_pu!=[0]:
				mag = float(find_magnitude(bottom_pu))
				if mag!=0.0:
					non_zero_mv+=1
			
			if non_zero_mv<2: # set it to 0 as noise MV
				copy_spatial_mv_data[frame, i, j] = str([0])
			else:
				mag_1 = find_magnitude(literal_eval(org_spatial_mv_data[frame-1,i,j]))
				mag_2 = find_magnitude(literal_eval(org_spatial_mv_data[frame+1,i,j]))

				if float(mag_1)==0.0 or float(mag_2)==0.0: # set it to 0 as noise MV
					copy_spatial_mv_data[frame, i, j] = str([0])

end_time = time.time()
print "Time taken: ", str(int(end_time-start_time)), " seconds"
np.save('./../'+video_type+'/'+video_name+'/output/'+'pic_mv_refining.npy',copy_spatial_mv_data)