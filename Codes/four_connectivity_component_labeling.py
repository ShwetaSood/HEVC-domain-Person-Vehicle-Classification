import os
import cv2
import sys
import json
import math
import time
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

spatial_mv_data = np.load('./../'+video_type+'/'+video_name+'/output/'+'pic_small_mv_removal.npy')
[frames ,height, width] = spatial_mv_data.shape
foreground_arr = np.zeros(shape=[frames ,height, width], dtype= int)

def connected(i,j):
	left = 1
	top = 1
	count = 0

	if i==0:
		top = 0
	if j==0:
		left = 0

	if left:
		if foreground_arr[frame, i, j-1]>=1:
			count+=1
	if top:
		if foreground_arr[frame, i-1, j]>=1:
			count+=1

	if count==0:
		return 0
	elif count==2:
		return 2
	elif count==1:
		if foreground_arr[frame, i, j-1]>=1:
			return -1
		elif foreground_arr[frame, i-1, j]>=1:
			return -2

def find_magnitude(mv_array):
	if len(mv_array)==1: #Intra coded => No MV
		return 0
	elif len(mv_array)==4 or len(mv_array)==2:
		return math.sqrt(math.pow(mv_array[-1],2)+math.pow(mv_array[-2],2))

start_time = time.time()
start_frame_no = 1
for frame in range(start_frame_no, no_frames):
	for i in range(len(spatial_mv_data[frame])):
		for j in range(len(spatial_mv_data[frame,0,:])):
			#print "Mag: ", find_magnitude(literal_eval(spatial_mv_data[frame,i,j]))
			if float(find_magnitude(literal_eval(spatial_mv_data[frame,i,j])))!=0.0:
				#print "Yes", frame, " ", i, " ", j
				foreground_arr[frame, i, j] = 1

for frame in range(start_frame_no, no_frames):
	cc = 0
	conflict = {}
	for i in range(len(foreground_arr[frame])):
		for j in range(len(foreground_arr[frame,0,:])):

			if foreground_arr[frame, i, j]==1:
				connected_val = connected(i,j)
				# if i==4 and j==49:
				#  	print "Reacheeeee ", connected_val, connected_val==0
				if connected_val==0:
					cc +=1
					foreground_arr[frame, i, j] = cc
				elif connected_val==2:
					if foreground_arr[frame, i-1, j] == foreground_arr[frame, i, j-1]:
						foreground_arr[frame, i, j] = foreground_arr[frame, i-1, j]
					else:
						foreground_arr[frame, i, j] = min(foreground_arr[frame, i-1, j],\
													  foreground_arr[frame, i, j-1])
						conflict[i-1, j] = [i, j-1]

				elif connected_val==-1:
					foreground_arr[frame, i, j] = foreground_arr[frame, i, j-1] # propagating left
				elif connected_val==-2:
					foreground_arr[frame, i, j] = foreground_arr[frame, i-1, j] # propagating top

	for key in conflict.keys():
		x = key[0]
		y = key[1]
		val_x = conflict[key][0]
		val_y = conflict[key][1]
		first = foreground_arr[frame, x, y]
		second = foreground_arr[frame, val_x, val_y]
		val_to_replace = max(first, second)
		foreground_arr[frame][foreground_arr[frame]==val_to_replace] = min(first, second)

	print "Frame ", frame, " After resolving conflict", np.count_nonzero(np.unique(foreground_arr[frame]))

end_time = time.time()
print "Time taken: ", str(int(end_time-start_time)), " seconds"
np.save('./../'+video_type+'/'+video_name+'/output/'+'foreground.npy', foreground_arr)

