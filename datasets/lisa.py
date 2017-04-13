# Import necessary libraries
import os, sys, shutil, glob, argparse
import numpy as np
from PIL import Image
import csv, ntpath

def make_directories(dataset):
	"""
	Definition: Make directories for kitti images and labels.
		Removes previously create kitti image and label directories.

	Parameters: dataset - path to {kitti, yolo, etc.} directory to be created
	Returns: None
	"""
	if os.path.exists(dataset):
		prompt = input("Directory already exists. Overwrite? (yes, no): ")
		if prompt == 'no':
			exit(0)
		shutil.rmtree(dataset)
	os.makedirs(dataset)
	os.makedirs(dataset + "train")
	os.makedirs(dataset + "train/images")
	os.makedirs(dataset + "train/labels")
	os.makedirs(dataset + "val")
	os.makedirs(dataset + "val/images")
	os.makedirs(dataset + "val/labels")

def split_train_val(data_split):
	"""
	Definition: Split the csv files for training and testing

	Parameters: data_split - split for data (i.e. '80' splits 80/20)
	Return: None
	"""
	os.system("python " + viva_signs + "scripts/filterAnnotationFile.py " +
		data_split + " " + lisa_annotations)
	os.system("python " + viva_signs + "scripts/filterAnnotationFile.py " +
		data_split + " " + lisa_ext_annotations)

###########################################################
##########        LISA to KITTI Conversion       ##########
###########################################################
def create_labels_kitti(fp_train, fp_val, out_train, out_val):
	f_train = open(fp_train, "r")
	f_val = open(fp_val, "r")

	csvf_train = csv.reader(f_train, delimiter=';')
	csvf_val = csv.reader(f_val, delimiter=';')
	header_train = csvf_train.next()
	header_val = csvf_val.next()

	fnameIdx_train = header_train.index("Filename")
	fnameIdx_val = header_val.index("Filename")
	tagIdx_train = header_train.index("Annotation tag")
	tagIdx_val = header_val.index("Annotation tag")
	upleftXIdx_train = header_train.index("Upper left corner X")
	upleftXIdx_val = header_val.index("Upper left corner X")
	upleftYIdx_train = header_train.index("Upper left corner Y")
	upleftYIdx_val = header_val.index("Upper left corner Y")
	lowrightXIdx_train = header_train.index("Lower right corner X")
	lowrightXIdx_val = header_val.index("Lower right corner X")
	lowrightYIdx_train = header_train.index("Lower right corner Y")
	lowrightYIdx_val = header_val.index("Lower right corner Y")

	for row in csvf_train:
		fname = ntpath.basename(row[fnameIdx_train])
		with open(out_train + fname + ".txt", 'a') as file:
			file.write('%s 0 0 0 %s %s %s %s 0 0 0 0 0 0 0 0\n'
				% (row[tagIdx_train], row[upleftXIdx_train], row[upleftYIdx_train],
					row[lowrightXIdx_train], row[lowrightYIdx_train]))

	for row in csvf_val:
		fname = ntpath.basename(row[fnameIdx_val])
		with open(out_val + fname + ".txt", 'a') as file:
			file.write('%s 0 0 0 %s %s %s %s 0 0 0 0 0 0 0 0\n'
				% (row[tagIdx_val], row[upleftXIdx_val], row[upleftYIdx_val],
					row[lowrightXIdx_val], row[lowrightYIdx_val]))

def kitti(lisa_dir, kitti_dir, label=None):
	print ("Converting lisa to kitti")

	# Make all directories for kitti dataset
	make_directories(kitti_dir)

	# Split into training and validation
	#split_train_val(70)

	# Write labels

	# Convert png.txt to .txt

	# Copy all images
	pass

###########################################################
##########        LISA to YOLO Conversion        ##########
###########################################################
def yolo(lisa_dir, yolo_dir, label=None):
	print ("Converting lisa to yolo")

	# Make all directories for yolo dataset
	make_directories(yolo_dir)
	pass

###########################################################
##########         LISA to VOC Conversion        ##########
###########################################################
def voc(lisa_dir, voc_dir, label=None):
	print ("Converting lisa to voc")

	# Make all directories for voc dataset
	make_directories(voc_dir)
	pass