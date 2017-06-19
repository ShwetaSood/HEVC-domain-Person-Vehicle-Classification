'''
Prepares training features for kmeans
'''

import os
import cv2
import sys
import math
import json
import time
import numpy as np
from ast import literal_eval

def find_magnitude(mv_array):
	if len(mv_array)==1: #Intra coded => No MV
		return 0
	elif len(mv_array)==4 or len(mv_array)==2:
		return math.sqrt(math.pow(mv_array[-1],2)+math.pow(mv_array[-2],2))

def mvd(spatial_mv_data, frame, x, y):
	if len(literal_eval(spatial_mv_data[frame, x, y]))==1:
		return [0, 0]

	left = 1
	right = 1
	top = 1
	bottom = 1
	top_left = 1
	top_right = 1
	bottom_right = 1
	bottom_left = 1

	if x==0:
		top = 0
		top_left = 0
		top_right = 0
	if y==0:
		left = 0
		top_left = 0
		bottom_left= 0 
	if x==(len(spatial_mv_data[frame])-1):
		bottom = 0
		bottom_left = 0
		bottom_right = 0
	if y==(len(spatial_mv_data[frame,0,:])-1):
		right = 0
		top_right = 0
		bottom_right = 0

	if left:
		left_pu = literal_eval(spatial_mv_data[frame,x,y-1])
	else:
		left_pu = [0]
	if right:
		right_pu = literal_eval(spatial_mv_data[frame,x,y+1])
	else:
		right_pu = [0]
	if top:
		top_pu = literal_eval(spatial_mv_data[frame,x-1,y])
	else:
		top_pu = [0]
	if bottom:
		bottom_pu = literal_eval(spatial_mv_data[frame,x+1,y])
	else:
		bottom_pu = [0]
	if top_left:
		top_left_pu = literal_eval(spatial_mv_data[frame, x-1, y-1])
	else:
		top_left_pu = [0]
	if top_right:
		top_right_pu = literal_eval(spatial_mv_data[frame, x-1, y+1])
	else:
		top_right_pu = [0]
	if bottom_left:
		bottom_left_pu = literal_eval(spatial_mv_data[frame, x+1, y-1])
	else:
		bottom_left_pu = [0]
	if bottom_right:
		bottom_right_pu = literal_eval(spatial_mv_data[frame, x+1, y+1])		
	else:
		bottom_right_pu = [0]

	neighborhood = [top_pu, top_left_pu, top_right_pu, \
	left_pu, right_pu, bottom_pu, bottom_left_pu, bottom_right_pu]
	max_curr_mvd = [0, 0]
	max_mag = 0

	curr_mv = literal_eval(spatial_mv_data[frame, x, y])
	for pu in neighborhood:
		if len(pu)==1:
			continue
		else:
			mag_curr_mvd = find_magnitude([abs(pu[-2]-curr_mv[-2]), abs(pu[-1]-curr_mv[-1])])
		if mag_curr_mvd>max_mag:
			max_mag = mag_curr_mvd
			max_curr_mvd = [abs(pu[-2]-curr_mv[-2]), abs(pu[-1]-curr_mv[-1])]

	return max_curr_mvd

def create_feature_vector(no_frames, prediction_data, spatial_mv_data, frame, x, y):
	prediction_mode = prediction_data[frame, x, y]
	if len(literal_eval(spatial_mv_data[frame, x, y]))==1:
		mv = [0,0]

	elif len(literal_eval(spatial_mv_data[frame, x, y]))==2 or\
		 len(literal_eval(spatial_mv_data[frame, x, y]))==4:
		 mv = [literal_eval(spatial_mv_data[frame, x, y])[-2], \
		       literal_eval(spatial_mv_data[frame, x, y])[-1]]

	max_mvd_curr_frame = mvd(spatial_mv_data, frame, x, y)
	if frame-1>=0:
		max_mvd_prev_frame = mvd(spatial_mv_data, frame-1, x, y)
	else:
		print "Range Error"
		raise SystemExit
	if frame+1<no_frames:
		max_mvd_fwd_frame = mvd(spatial_mv_data, frame+1, x, y)
	else:
		print "Range Error"
		raise SystemExit

	return [int(prediction_mode), int(mv[-2]), int(mv[-1]), \
			int(max_mvd_curr_frame[-2]), int(max_mvd_curr_frame[-1]), \
			int(max_mvd_prev_frame[-2]), int(max_mvd_prev_frame[-1]), \
			int(max_mvd_fwd_frame[-2]), int(max_mvd_fwd_frame[-1])]

if __name__ == "__main__":

	'''
	input parameters
	'''
	#-----------------------------------------------

	video_type = 'Pedestrian_dataset' # Vehicle_dataset, Pedestrian_dataset
	video_name = 'hall_monitor'

	#-----------------------------------------------

	with open('./../Config/video_config.json') as data_file:
		video_data = json.load(data_file)
	video_width = video_data[video_type][video_name]["width"]
	video_height = video_data[video_type][video_name]["height"]
	no_frames=  video_data[video_type][video_name]["no_frames"]

	foreground_arr = np.load('./../'+video_type+'/'+video_name+'/output/'+'pic_object_boundary_refinement.npy')
	spatial_mv_data = np.load('./../'+video_type+'/'+video_name+'/output/'+'pic_small_mv_removal.npy')
	prediction_data = np.load('./../'+video_type+'/'+video_name+'/output/'+'pic_pred.npy')

	start_frame_no = 1
	end_frame_no = no_frames - 1

	feature_set = []
	start_time = time.time()
	for frame in range(start_frame_no, end_frame_no): #HAVE TO ASK RANGE
		print frame
		curr_frame_unique_foreground_values = np.delete(np.unique(foreground_arr[frame]),0)

		for val in curr_frame_unique_foreground_values: # Iterates over all foreground regions
			X,Y=np.where(foreground_arr[frame]==val) # Locations of a particular foreground region
			for i in range(len(X)):
				feature_vector = create_feature_vector(no_frames, \
				prediction_data, spatial_mv_data, frame, X[i], Y[i])
				feature_set+=[feature_vector]

	end_time = time.time()
	print "Time taken: ", str(int(end_time-start_time)), " seconds"
	
	try:
		old_features = np.load('./../Standalone/features.npy')
		feature_set = np.concatenate((old_features, np.array(feature_set)))
		np.save('./../Standalone/features.npy', feature_set)
	except:
		np.save('./../Standalone/features.npy', np.array(feature_set))