'''
Use the old version - old_object_region_tracking.py
Also, if you feel like comparing a new video's output for the 2 versions
set : for fore_temporal_val in range(0,4): to range(0,3) AND
set : for temporal_val in range(0,4): to range (0,3)
'''

import os
import cv2
import sys
import math
import json
import operator
import numpy as np
from ast import literal_eval

'''
input parameters
'''
#-----------------------------------------------

video_type = 'Pedestrian_dataset' # Vehicle_dataset, Pedestrian_dataset
video_name = 'hall_monitor'

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

spatial_mv_data = np.load('./../'+video_type+'/'+video_name+'/output/'+'pic_small_mv_removal.npy')
foreground_arr = np.load('./../'+video_type+'/'+video_name+'/output/'+'foreground.npy')
copy_foreground_arr = np.load('./../'+video_type+'/'+video_name+'/output/'+'foreground.npy')
forward_projection_arr = np.load('./../'+video_type+'/'+video_name+'/output/'+'forward_projection.npy')

[frames ,height, width] = spatial_mv_data.shape

start_frame_no = 4
end_frame_no = no_frames - 4

def find_magnitude(mv_array):
	if len(mv_array)==1: #Intra coded => No MV
		return 0
	elif len(mv_array)==4 or len(mv_array)==2:
		return math.sqrt(math.pow(mv_array[-1],2)+math.pow(mv_array[-2],2))

# BACKGROUND
for frame in range(start_frame_no, end_frame_no):
	print frame
	curr_frame_unique_foreground_values = np.delete(np.unique(foreground_arr[frame]),0)

	for val in curr_frame_unique_foreground_values: # Iterates over all foreground regions
		X,Y=np.where(foreground_arr[frame]==val) # Locations of a particular foreground region
		temp_X = X
		temp_Y = Y
		btemporal = True
		for temporal_val in range(0,4): # Evaluating temporal consistency of a particular foreground block
			no_of_projections = {}
			temp_curr_frame = frame - temporal_val
			for i in range(len(temp_X)): # Iterates over all 4x4 blocks in a particular foreground region

				horz_disp = int(literal_eval\
							(spatial_mv_data[temp_curr_frame, temp_X[i], temp_Y[i]])[-2])
				vert_disp = int(literal_eval\
							(spatial_mv_data[temp_curr_frame, temp_X[i], temp_Y[i]])[-1])


				projected_x = (temp_X[i]*4 + (np.sign(vert_disp)*(abs(vert_disp)/4)))/4 # row + vertical disp
				projected_y = (temp_Y[i]*4 + (np.sign(horz_disp)*(abs(horz_disp)/4)))/4 # col + horizontal disp

				if foreground_arr[temp_curr_frame-1, projected_x, projected_y]!=0:
					if no_of_projections.get(foreground_arr[temp_curr_frame-1, projected_x, projected_y]):
						no_of_projections[foreground_arr[temp_curr_frame-1, projected_x, projected_y]]+=1
					else:
						no_of_projections[foreground_arr[temp_curr_frame-1, projected_x, projected_y]] = 1
			if no_of_projections:
				foreground_region_no = max(no_of_projections.iteritems(), key=operator.itemgetter(1))[0]
				if float(no_of_projections[foreground_region_no])/float(len(temp_X))<=0.5:
					btemporal = False
					break
			else:
				btemporal = False
				break
			temp_X,temp_Y=np.where(foreground_arr[temp_curr_frame-1]==foreground_region_no)
		if not btemporal:
			for iteration in range(len(X)):
				copy_foreground_arr[frame, X[iteration], Y[iteration]] = 0		
		else:

			X,Y=np.where(foreground_arr[frame]==val) # Locations of a particular foreground region
			fore_temp_X = X
			fore_temp_Y = Y

			for fore_temporal_val in range(0,4): # Evaluating temporal consistency of a particular foreground block
				no_of_projections = {}
				fore_temp_curr_frame = frame + fore_temporal_val

				for i in range(len(fore_temp_X)): # Iterates over all 4x4 blocks in a particular foreground region

					if forward_projection_arr[fore_temp_curr_frame, fore_temp_X[i], fore_temp_Y[i]]!='':

						projected_x = literal_eval\
						(forward_projection_arr[fore_temp_curr_frame, fore_temp_X[i], fore_temp_Y[i]])[-2]
						projected_y = literal_eval\
						(forward_projection_arr[fore_temp_curr_frame, fore_temp_X[i], fore_temp_Y[i]])[-1]

						if no_of_projections.get(foreground_arr[fore_temp_curr_frame+1, projected_x, projected_y]):
							no_of_projections[foreground_arr[fore_temp_curr_frame+1, projected_x, projected_y]]+=1
						else:
							no_of_projections[foreground_arr[fore_temp_curr_frame+1, projected_x, projected_y]] = 1

				if no_of_projections:
					foreground_region_no = max(no_of_projections.iteritems(), key=operator.itemgetter(1))[0]
					if float(no_of_projections[foreground_region_no])/float(len(fore_temp_X))<=0.5:
						btemporal = False
						break
				else:
					btemporal = False
					break
				fore_temp_X,fore_temp_Y=np.where(foreground_arr[fore_temp_curr_frame+1]==foreground_region_no)
			if not btemporal:
				for iteration in range(len(X)):
					copy_foreground_arr[frame, X[iteration], Y[iteration]] = 0
			else:
				print "Consistent"

np.save('./../'+video_type+'/'+video_name+'/output/'+'new_pic_object_track.npy',copy_foreground_arr)