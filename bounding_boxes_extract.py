import cv2
import json
import sys
from PIL import Image
import numpy as np
import os
import argparse

# crop bounding box, check, resize it, check,  overlap it, check
def arg_parse():
	parser = argparse.ArgumentParser(description = 'Bounding boxes')

	parser.add_argument('--mode', dest = 'mode', help = 'Testing or Training', type = str, default = 'train')

	return parser.parse_args()


def get_bounding_boxes():

	with open('train_data.json', 'r') as infile:
		data = json.load(infile)

	# for k in range(len(data)):

		img = cv2.imread('./sg_dataset/sg_train_images/' + str(data[3]['file_name']))

		print data[0]['file_name'], len(data[0]['boxes']), 0
		# for i in data[0]['boxes']:
		for b in range(len(data[3]['boxes'])):

			i = data[3]['boxes'][b]
			subject = img[ i[0][1] : i[0][1] + i[0][3], i[0][0] : i[0][0] + i[0][2] ]
			_object = img[ i[1][1] : i[1][2] + i[1][3], i[1][0] : i[1][0] + i[1][2] ]
			
			res = 0 if i[0][0] < i[1][0] else 1
			res2 = 1 if i[0][0] < i[1][0] else 0

			p_x = i[res][0]
			p_y = i[res][1]

			p_w = abs(i[res2][0] + i[res2][2] - i[res][0])
			p_h = abs(i[res2][1] + i[res2][3] - i[res][1])

			p_w = 100 if p_w == 0 else p_w
			p_h = 100 if p_h == 0 else p_h
			# print (i[0][0], i[0][1], i[0][2], i[0][3])
			# print (i[1][0], i[1][1], i[1][2], i[1][3])
			print (p_x, p_y, p_w, p_h)
			
			predicate = img[p_y : p_y + p_h, p_x : p_x + p_w]

			subject = cv2.resize(subject, (500, 500))
			_object = cv2.resize(_object, (500, 500))
			predicate = cv2.resize(predicate, (500, 500))


			subject = np.array(subject)
			predicate = np.array(predicate)
			_object = np.array(_object)

			img_merge = np.hstack((subject, predicate))
			img_merge = np.hstack((img_merge, _object))

			# img_merge = cv2.resize(img_merge, (32, 32))

			# cv2.imshow('subject', subject)
			# cv2.imshow('predicate', predicate)
			# cv2.imshow('object', _object)
			
			# image_name_save = str(data[k]['file_name'][:-4]) + '_box_' + str(b) + '.jpg'

			cv2.imshow('concatenated image', img_merge)	
			# cv2.imwrite('./relationship_images/' + str(image_name_save), img_merge)
			cv2.waitKey(0)

def get_bounding_boxes_test():

	with open('test_data.json', 'r') as infile:
		data = json.load(infile)

	l = []
	for _file in os.listdir("./sg_dataset/test"):
		l.append(_file)

	for k in l:
		img = cv2.imread('./sg_dataset/test/' + str(k))
		
		print data[0]['file_name'], len(data[0]['boxes']), 0
		# for i in data[0]['boxes']:
		for b in range(len(data[3]['boxes'])):

			i = data[3]['boxes'][b]
			subject = img[ i[0][1] : i[0][1] + i[0][3], i[0][0] : i[0][0] + i[0][2] ]
			_object = img[ i[1][1] : i[1][2] + i[1][3], i[1][0] : i[1][0] + i[1][2] ]
			
			res = 0 if i[0][0] < i[1][0] else 1
			res2 = 1 if i[0][0] < i[1][0] else 0

			p_x = i[res][0]
			p_y = i[res][1]

			p_w = abs(i[res2][0] + i[res2][2] - i[res][0])
			p_h = abs(i[res2][1] + i[res2][3] - i[res][1])

			p_w = 100 if p_w == 0 else p_w
			p_h = 100 if p_h == 0 else p_h
			# print (i[0][0], i[0][1], i[0][2], i[0][3])
			# print (i[1][0], i[1][1], i[1][2], i[1][3])
			print (p_x, p_y, p_w, p_h)
			
			predicate = img[p_y : p_y + p_h, p_x : p_x + p_w]

			subject = cv2.resize(subject, (500, 500))
			_object = cv2.resize(_object, (500, 500))
			predicate = cv2.resize(predicate, (500, 500))


			subject = np.array(subject)
			predicate = np.array(predicate)
			_object = np.array(_object)

			img_merge = np.hstack((subject, predicate))
			img_merge = np.hstack((img_merge, _object))

			# img_merge = cv2.resize(img_merge, (32, 32))

			# cv2.imshow('subject', subject)
			# cv2.imshow('predicate', predicate)
			# cv2.imshow('object', _object)
			
			image_name_save = str(k[:-4]) + '_box_' + str(b) + '.jpg'

			cv2.imshow('concatenated image', img_merge)	
			cv2.imwrite('./sg_dataset/visual_bb/' + str(image_name_save), img_merge)
			cv2.waitKey(0)



if __name__=='__main__':

	args = arg_parse()
	if args.mode == "test":
		get_bounding_boxes_test()
	else:
		get_bounding_boxes()