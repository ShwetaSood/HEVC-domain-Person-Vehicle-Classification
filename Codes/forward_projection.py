'''
Stores projection of t to t-1 in frame t
this code is only useful for 'old_region_tracking.py'
otherwise, don't use
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
video_name = 'atrium_1'

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
forward_projection_arr = np.chararray(shape=spatial_mv_data.shape, itemsize=100)
[frames ,height, width] = spatial_mv_data.shape

start_frame_no = 1
end_frame_no = no_frames

for frame in range(start_frame_no, end_frame_no):
	print frame
	curr_frame_unique_foreground_values = np.delete(np.unique(foreground_arr[frame]),0)

	for val in curr_frame_unique_foreground_values: # Iterates over all foreground regions
		X,Y=np.where(foreground_arr[frame]==val) # Locations of a particular foreground region

		no_of_projections = {}

		for i in range(len(X)): # Iterates over all 4x4 blocks in a particular foreground region

			horz_disp = int (literal_eval\
						(spatial_mv_data[frame, X[i], Y[i]])[-2])
			vert_disp = int(literal_eval\
						(spatial_mv_data[frame, X[i], Y[i]])[-1])

			projected_x = (X[i]*4 + (np.sign(vert_disp)*(abs(vert_disp)/4)))/4 # row + vertical disp
			projected_y = (Y[i]*4 + (np.sign(horz_disp)*(abs(horz_disp)/4)))/4 # col + horizontal disp

			try:
				assert projected_x>=0 and projected_x<height
				assert projected_y>=0 and projected_y<width
			except:
				continue
			forward_projection_arr[frame-1, projected_x, projected_y] = str([X[i], Y[i]])

np.save('./../'+video_type+'/'+video_name+'/output/'+'forward_projection.npy', forward_projection_arr)