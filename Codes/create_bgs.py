import os
import cv2
import json
import numpy as np

'''
input parameters
'''
#-----------------------------------------------

video_type = 'Pedestrian_dataset' # Vehicle_dataset, Pedestrian_dataset
video_name = 'data_0053'

#-----------------------------------------------

with open('./../Config/video_config.json') as data_file:
	video_data = json.load(data_file)

input_image_path = './../'+video_type+'/'+video_name+'/input'
input_images =  np.sort(os.listdir(input_image_path))

video_width = video_data[video_type][video_name]["width"]
video_height = video_data[video_type][video_name]["height"]
no_frames=  video_data[video_type][video_name]["no_frames"]

foreground_arr = np.load('./../'+video_type+'/'+video_name+'/output/'+'pic_object_boundary_refinement.npy')
bgs_frames = np.zeros(shape=[no_frames, video_height*4, video_width*4], dtype=np.int32)

start_frame_no = 0
end_frame_no = no_frames

for frame in range(start_frame_no, end_frame_no):
	print frame
	for i in range(len(foreground_arr[frame])):
		for j in range(len(foreground_arr[frame,0,:])):
			if foreground_arr[frame, i, j]!=0:
				print "enter"
				bgs_frames[frame, i:i+4, j:j+4] = 255
	# cv2.imwrite('./../'+video_type+'/'+video_name+'/bgs/'+input_images[frame][:-3]+'.png'\
	# 		    ,bgs_frames[frame])