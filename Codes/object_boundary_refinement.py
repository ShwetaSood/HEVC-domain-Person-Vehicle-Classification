import os
import cv2
import sys
import math
import json
import time
import operator
import numpy as np
from ast import literal_eval

'''
input parameters
'''
#-----------------------------------------------

video_type = 'Pedestrian_dataset' # Vehicle_dataset, Pedestrian_dataset
video_name = 'pedestrians'

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

depth_data = np.load('./../'+video_type+'/'+video_name+'/output/'+'pic_depth.npy')
foreground_arr = np.load('./../'+video_type+'/'+video_name+'/output/'+'pic_object_track.npy')
copy_foreground_arr = np.load('./../'+video_type+'/'+video_name+'/output/'+'pic_object_track.npy')

depth_cu_dict = {0:64,1:32,2:16,3:8,4:4}
start_frame_no = 1
end_frame_no = no_frames
[frames ,height, width] = depth_data.shape

def boundary_refinement(frame, foreground, copy_foreground, depth):
	[frames ,height, width] = depth.shape

	curr_frame_unique_foreground_values = np.delete(np.unique(foreground[frame]),0)

	for val in curr_frame_unique_foreground_values: # Iterates over all foreground regions
		total = 0
		count = 0
		X,Y=np.where(foreground[frame]==val) # Locations of a particular foreground region
		for i in range(len(X)):
			total+=depth[frame, X[i], Y[i]]
			count+=1

		avg_depth = math.floor((float(total)/count) + 0.5) - 1

		unique_X = np.unique(X)
		for value in unique_X: # to iterate over a row in a foreground region
			index = np.where(X==value)[0]
			maximum_depth = depth[frame, X[index[0]], Y[index[0]]]
			minimum_depth = depth[frame, X[index[0]], Y[index[0]]]
			pos_min_depth = Y[index[0]]

			for i in range(1, len(index)): # to iterate over a column to find candidate row
				if (depth[frame, X[index[i]], Y[index[i]]]) >  maximum_depth:
					maximum_depth = depth[frame, X[index[i]], Y[index[i]]]

				if (depth[frame, X[index[i]], Y[index[i]]]) <  minimum_depth:
					minimum_depth = depth[frame, X[index[i]], Y[index[i]]]
					pos_min_depth = Y[index[i]]
			
			if maximum_depth<=avg_depth:
				candidate_block_row = value
				index_block = pos_min_depth
				cu_size = depth_cu_dict.get(depth[frame, candidate_block_row, index_block])
				cu_start_row = (((candidate_block_row*4)/cu_size)*cu_size)/4
				cu_start_col = (((index_block*4)/cu_size)*cu_size)/4
				top_cu_start_row = cu_start_row-1
				top_cu_start_col = cu_start_col
				bottom_cu_start_row = cu_start_row+(cu_size/4)
				bottom_cu_start_col = cu_start_col

				if top_cu_start_row>=0:
					max_top_row = \
					max(depth[frame, top_cu_start_row, \
						top_cu_start_col : top_cu_start_col+(cu_size/4)])
					if max_top_row<=avg_depth:
						for i in range(len(index)): # setting background block as satisifes condition
							copy_foreground[frame, candidate_block_row, Y[index[i]]] = 0
						continue

				if bottom_cu_start_row<height:
					max_bottom_row = \
					max(depth[frame, bottom_cu_start_row, \
						bottom_cu_start_col : bottom_cu_start_col+(cu_size/4)])
					if max_bottom_row<=avg_depth:
						for i in range(len(index)): # setting background block as satisifes condition
							copy_foreground[frame, candidate_block_row, Y[index[i]]] = 0

	return copy_foreground

start_time = time.time()
for frame in range(start_frame_no, end_frame_no):
	print frame
	copy_foreground_arr = boundary_refinement(frame, foreground_arr, copy_foreground_arr, depth_data)
	copy_foreground_arr = (boundary_refinement(frame, foreground_arr.transpose(0,2,1), \
						  copy_foreground_arr.transpose(0,2,1), depth_data.transpose(0,2,1))).transpose(0,2,1)

end_time = time.time()
print "Time taken: ", str(end_time-start_time), " seconds"
np.save('./../'+video_type+'/'+video_name+'/output/'+'pic_object_boundary_refinement.npy',copy_foreground_arr)