import os
import cv2
import sys
import math
import json
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

video_width = video_data[video_type][video_name]["width"]
video_height = video_data[video_type][video_name]["height"]
no_frames=  video_data[video_type][video_name]["no_frames"]

input_image_path = './../'+video_type+'/'+video_name+'/input/'
input_images =  np.sort(os.listdir(input_image_path))

#spatial_mv_data = np.load('./../'+video_type+'/'+video_name+'/output/'+'pic_small_mv_removal.npy')
#spatial_mv_data = np.load('./../'+video_type+'/'+video_name+'/output/'+'pic_mv.npy')
#spatial_mv_data = np.load('./../'+video_type+'/'+video_name+'/output/'+'pic_mv_filtering.npy')
#spatial_mv_data = np.load('./../'+video_type+'/'+video_name+'/output/'+'pic_mv_refining.npy')
#spatial_mv_data = np.load('pic_object_track.npy')
spatial_mv_data = np.load('./../'+video_type+'/'+video_name+'/output/'+'pic_object_boundary_refinement.npy')
#spatial_mv_data = np.load('./../'+video_type+'/'+video_name+'/output/'+'pic_object_track.npy')
#spatial_mv_data = np.load('./../'+video_type+'/'+video_name+'/output/'+'foreground.npy')


[frames ,height, width] = spatial_mv_data.shape

def find_magnitude(mv_array):
	if len(mv_array)==1: #Intra coded => No MV
		return 0
	elif len(mv_array)==4 or len(mv_array)==2:
		return math.sqrt(math.pow(mv_array[-1],2)+math.pow(mv_array[-2],2))

start_frame_no = 0 # First frame is I frame
curr_frame_no = start_frame_no

for frame in range(start_frame_no, no_frames):
	print curr_frame_no
	img = cv2.imread(input_image_path+input_images[curr_frame_no], cv2.IMREAD_COLOR)

	for i in range(len(spatial_mv_data[frame])):
		for j in range(len(spatial_mv_data[frame,0,:])):
			#if float(find_magnitude(literal_eval(spatial_mv_data[frame, i, j])))!=0.0:
			if spatial_mv_data[frame, i, j]!=0:
				temp_start_w = 4*j
				temp_start_h = 4*i
				cv2.line(img,(temp_start_w, temp_start_h),(4+temp_start_w, temp_start_h),(255,255,255),1)
				cv2.line(img,(temp_start_w,temp_start_h),(temp_start_w,4+temp_start_h),(255,255,255),1)
				cv2.line(img,(4+temp_start_w, temp_start_h),(4+temp_start_w, 4+temp_start_h),(255,255,255),1)
				cv2.line(img,(temp_start_w,4+temp_start_h),(4+temp_start_w, 4+temp_start_h),(255,255,255),1)
				# cv2.imshow('image',img)
				# cv2.waitKey(0)
				# cv2.destroyAllWindows()
		cv2.imwrite('./../'+video_type+'/'+video_name+'/photos/'+'bound'+ str(frame) + ".jpg", img)
	curr_frame_no+=1