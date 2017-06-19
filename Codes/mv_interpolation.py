import os
import cv2
import sys
import math
import json
import time
import numpy as np
import pandas as pd
from ast import literal_eval

'''
input parameters
'''
#-----------------------------------------------

video_type = 'Vehicle_dataset' # Vehicle_dataset, Pedestrian_dataset
video_name = 'sherbrooke_4'

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

with open(filename+'/decoder_cupu.txt') as f1:
	data = f1.readlines()
with open(filename+'/decoder_pred.txt') as f2:
	pred_data = f2.readlines()
with open(filename+'/decoder_mv.txt') as f3:
	mv_data = f3.readlines()

spatial_mv_data = np.load('./../'+video_type+'/'+video_name+'/output/'+'pic_mv.npy')
copy_spatial_mv_data = np.load('./../'+video_type+'/'+video_name+'/output/'+'pic_mv.npy')

[frames ,height, width] = spatial_mv_data.shape

def rescaling(frame_no):
	for i in range(len(spatial_mv_data[frame_no])):
		for j in range(len(spatial_mv_data[frame_no,0,:])):
			mv_vector = literal_eval(spatial_mv_data[frame_no, i, j])
			if len(mv_vector)==4:
				scaling_factor = abs(frame_no - mv_vector[1])
				mv_vector[2] = mv_vector[2]/float(scaling_factor)
				mv_vector[3] = mv_vector[3]/float(scaling_factor)
				copy_spatial_mv_data[frame_no,i,j] = str(mv_vector) #normalization

def find_magnitude(mv_array):
	if len(mv_array)==1: #Intra coded => No MV
		return 0
	elif len(mv_array)==4:
		return math.sqrt(math.pow(mv_array[-1],2)+math.pow(mv_array[-2],2))

def interpolation(frame_no, temp_start_h, temp_start_w, cu_height, cu_width):

	if temp_start_h-1>0:
		x_start = temp_start_h-1
	else:
		x_start = temp_start_h
	if temp_start_w-1>0:
		y_start = temp_start_w-1
	else:
		y_start = temp_start_w
	if temp_start_h+cu_height<height:
		x_end = temp_start_h+cu_height
	else:
		x_end = temp_start_h+cu_height-1
	if temp_start_w+cu_width<width:
		y_end = temp_start_w+cu_width
	else:
		y_end = temp_start_w+cu_width-1

	neighborhood_mv = np.concatenate((spatial_mv_data[frame_no, x_start:x_end+1,y_start],\
	spatial_mv_data[frame_no, x_start, y_start:y_end+1],\
	spatial_mv_data[frame_no, x_end, y_start:y_end+1],\
	spatial_mv_data[frame_no, x_start:x_end+1, y_end]))
	maximum_mv = 0
	pos = -1
	for index in range(len(neighborhood_mv)):
		curr_mag_mv = find_magnitude(literal_eval(neighborhood_mv[index]))
		if curr_mag_mv>maximum_mv:
			maximum_mv = curr_mag_mv
			pos = index
	if pos==-1:
		if temp_start_h+cu_height+3<height: # empirical value of 16
			x_end = temp_start_h+cu_height+3
		else:
			x_end = height-1
		if temp_start_w+cu_width+3<width:
			y_end = temp_start_w+cu_width+3
		else:
			y_end = width-1

			neighborhood_mv = np.concatenate((spatial_mv_data[frame_no, x_start:x_end+1,y_start],\
			spatial_mv_data[frame_no, x_start, y_start:y_end+1],\
			spatial_mv_data[frame_no, x_end, y_start:y_end+1],\
			spatial_mv_data[frame_no, x_start:x_end+1, y_end]))
			maximum_mv = 0
			pos = -1
			for index in range(len(neighborhood_mv)):
				curr_mag_mv = find_magnitude(literal_eval(neighborhood_mv[index]))
				if curr_mag_mv>maximum_mv:
					maximum_mv = curr_mag_mv
					pos = index
			if pos==-1:
				return
			else:
				copy_spatial_mv_data[frame_no, temp_start_h:temp_start_h+cu_height, \
				temp_start_w:temp_start_w+cu_width] = str([literal_eval(neighborhood_mv[pos])[-2],\
													  literal_eval(neighborhood_mv[pos])[-1]])
	else:
		copy_spatial_mv_data[frame_no, temp_start_h:temp_start_h+cu_height, \
		temp_start_w:temp_start_w+cu_width] = str([literal_eval(neighborhood_mv[pos])[-2],\
											  literal_eval(neighborhood_mv[pos])[-1]])

def decode_cupu(pos_cupu, cupu_list, pred_list, mv_list, start_w, start_h, \
	cu_width, cu_height, cu_depth):
	'''
	pos_cupu: position in cupu_list
	'''
	global pos_mv
	global pos_pred
	global frame_no
	count = 0
	i = pos_cupu
	while i<len(cupu_list):
		if count==0:
			temp_start_w = start_w
			temp_start_h = start_h
		elif count==1:
			temp_start_w = start_w + cu_width
			temp_start_h = start_h
		elif count==2:
			temp_start_w = start_w
			temp_start_h = start_h + cu_height
		elif count==3:
			temp_start_w = start_w + cu_width
			temp_start_h = start_h + cu_height
		elif count==4:
			return i-1

		if cupu_list[i]=='99':
			# print "Achieve     ", cupu_list
			i = decode_cupu(i+1, cupu_list, pred_list, mv_list, temp_start_w, temp_start_h, \
				cu_width/2, cu_height/2, cu_depth+1)
			count+=1
		elif cupu_list[i]=='0':

			if pred_list[pos_pred]==2 and frame_no>0: # Code for MV interpolation
				interpolation(frame_no, temp_start_h/4, temp_start_w/4, cu_height/4,\
					cu_width/4)
			pos_pred += 1
			count+=1

		elif cupu_list[i]=='3': # Entering a PU

			if pred_list[pos_pred]==2 and frame_no>0: # Code for MV interpolation
				interpolation(frame_no, temp_start_h/4, temp_start_w/4, cu_height/8,\
					cu_width/8)

			if pred_list[pos_pred+1]==2 and frame_no>0: # Code for MV interpolation
				interpolation(frame_no, temp_start_h, (temp_start_w)/4+(cu_width)/8,\
					cu_height/8, cu_width/8)

			if pred_list[pos_pred+2]==2 and frame_no>0: # Code for MV interpolation
				interpolation(frame_no, (temp_start_h)/4+(cu_height)/8, temp_start_w,\
					cu_height/8, cu_width/8)

			if pred_list[pos_pred+3]==2 and frame_no>0: # Code for MV interpolation
				interpolation(frame_no, (temp_start_h)/4+(cu_height)/8, (temp_start_w)/4+(cu_width)/8,\
					cu_height/8, cu_width/8)

			pos_pred += 4

			count+=1
		elif cupu_list[i]=='15':
			count+=1
		i+=1

	return i

start_time = time.time()

for frame in range(no_frames):
	rescaling(frame)

for index in range(len(data)):

	cupu_info, cupu_struct= data[index].split('> ')
	pred_list = pred_data[index].split('> ')[1].strip('\n\r ').split(' ')
	pred_list = [int(el) for el in pred_list]
	mv_list = mv_data[index].split('> ')[1].strip('\n\r ').split(' ')
	mv_list = [int(el) for el in mv_list]
	cupu_list = cupu_struct.strip('\n\r ').split(' ')
	frame_no, ctu_no= cupu_info.strip('<').split(',')
	frame_no = int(frame_no)

	if frame_no==0 and int(ctu_no)==0: # reading the image for first time
		print frame_no
		curr_frame_no = 0
		img = cv2.imread(input_image_path+input_images[curr_frame_no], cv2.IMREAD_COLOR)

	if frame_no!=curr_frame_no:
		print frame_no
		curr_frame_no = frame_no
		img = cv2.imread(input_image_path+input_images[curr_frame_no], cv2.IMREAD_COLOR)
		
	pos_pred = 0
	pos_mv = 0

	decode_cupu(0, cupu_list, pred_list, mv_list, \
		64*(int(ctu_no)%no_ctu_width), 64*(int(ctu_no)/no_ctu_width), 64, 64, 0)

end_time = time.time()
print "Time taken: ", str(int(end_time-start_time)), " seconds"
np.save('./../'+video_type+'/'+video_name+'/output/'+'pic_mv_interpolated.npy',copy_spatial_mv_data)