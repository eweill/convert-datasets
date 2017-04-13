###############################################################################
##########                        YOLO format                        ##########
"""
Values    Name      Description
----------------------------------------------------------------------------
   1    type         Describes the type of object by the index value from the
                     label file for each class (i.e. 0=Car, 1=Pedestrian)
   4    bbox         2D bounding box of object in the image (0-based index):
                     contains x, y, width, height where x and y are the center
                     of the image and all coordinates are normalized based on
                     the image size
"""
###############################################################################

# Import necessary libraries
import os, sys, shutil, glob, argparse
import numpy as np
from PIL import Image

###########################################################
##########       YOLO to KITTI Conversion        ##########
###########################################################
def determine_label_kitti(label, labels):
	"""
	Definition: Converts label index to label from label set

	Parameters: label - label from file
				labels - list of labels
	Returns: label corresponding to index in yolo label
	"""
	return str(labels[int(label)])

def parse_labels_kitti(label_file, labels, img_width, img_height):
	"""
	Definition: Parsers label file to extract label and bounding box
		coordinates. Converts (x, y, width, height) YOLO format to
		(x1, y1, x2, y2) KITTI format.

	Parameters: label_file - file with YOLO label(s) inside
				labels - list of labels in dataset
				img_width - width of input image
				img_height - height of input image
	Returns: all_labels - contains a list of labels for objects in image
			 all_coords - contains a list of coordinates for objects in image
	"""
	lfile = open(label_file)
	coords = []
	all_coords = []
	all_labels = []
	for line in lfile:
		l = line.split(" ")
		all_labels.append(determine_label_kitti(l[0], labels))
		coords = list(map(float, list(map(float, l[1:5]))))
		x1 = float(img_width) * (2.0 * float(coords[0]) - float(coords[2])) / 2.0
		y1 = float(img_height) * (2.0 * float(coords[1]) - float(coords[3])) / 2.0
		x2 = float(img_width) * (2.0 * float(coords[0]) + float(coords[2])) / 2.0
		y2 = float(img_height) * (2.0 * float(coords[1]) + float(coords[3])) / 2.0
		tmp = [x1, y1, x2, y2]
		all_coords.append(list(map(int, tmp)))
	lfile.close()
	return all_labels, all_coords

def copy_images_kitti(yolo, kitti):
	"""
	Definition: Copy all images from the training and validation sets
		in yolo format to training and validation image sets in kitti
		format.  This means converting from .jpg to .png

	Parameters: yolo - path to yolo directory (contains 'train' and 'val')
				kitti - path to kitti output directory
	Returns: None
	"""
	for filename in glob.glob(os.path.join(yolo + "train/images/", "*.*")):
		shutil.copy(filename, kitti + "train/images/")
	for filename in glob.glob(os.path.join(yolo + "val/images/", "*.*")):
		shutil.copy(filename, kitti + "val/images/")

	for filename in glob.glob(os.path.join(kitti + "train/images/", "*.*")):
		im = Image.open(filename)
		im.save(filename.split(".jpg")[0] + ".png", "png")
		os.remove(filename)
	for filename in glob.glob(os.path.join(kitti + "val/images/", "*.*")):
		im = Image.open(filename)
		im.save(filename.split(".jpg")[0] + ".png", "png")
		os.remove(filename)

def make_kitti_directories(kitti):
	"""
	Definition: Make directories for kitti images and labels.
		Removes previously created kitti image and label directories.

	Parameters: kitti - path to kitti directory to be created
	Returns: None
	"""
	if os.path.exists(kitti):
		prompt = input('Directory already exists. Overwrite? (yes, no): ')
		if prompt == 'no':
			exit(0)
		shutil.rmtree(kitti)
	os.makedirs(kitti)
	os.makedirs(kitti + "train")
	os.makedirs(kitti + "train/images")
	os.makedirs(kitti + "train/labels")
	os.makedirs(kitti + "val")
	os.makedirs(kitti + "val/images")
	os.makedirs(kitti + "val/labels")

def kitti(yolo_dir, kitti_dir, label=None):
	print ("Converting yolo to kitti")

	# Split label file
	label_file = open(label)
	labels_split = label_file.read().split('\n')

	# Make all directories for kitti dataset
	make_kitti_directories(kitti_dir)

	# Iterate through yolo training data
	for f in os.listdir(yolo_dir + "train/labels/"):
		fname = (yolo_dir + "train/images/" + f).split(".txt")[0] + ".jpg"
		if os.path.isfile(fname):
			img = Image.open(fname)
			w, h = img.size
			img.close()
			labels, coords = parse_labels_kitti(os.path.join(yolo_dir + 
				"train/labels/" + f), labels_split, w, h)
			yolof = open(kitti_dir + "train/labels/" + f, "a+")
			for l, c in zip(labels, coords):
				yolof.write(l + " 0 0 0 " + str(c[0]) + " " + str(c[1]) +
					" " + str(c[2]) + " " + str(c[3]) + " 0 0 0 0 0 0 0 0\n")
			yolof.close()

	# Iterate through yolo validation data
	for f in os.listdir(yolo_dir + "val/labels/"):
		fname = (yolo_dir + "val/images/" + f).split(".txt")[0] + ".jpg"
		if os.path.isfile(fname):
			img = Image.open(fname)
			w, h = img.size
			img.close()
			labels, coords = parse_labels_kitti(os.path.join(yolo_dir + 
				"val/labels/" + f), labels_split, w, h)
			yolof = open(kitti_dir + "val/labels/" + f, "a+")
			for l, c in zip(labels, coords):
				yolof.write(l + " 0 0 0 " + str(c[0]) + " " + str(c[1]) +
					" " + str(c[2]) + " " + str(c[3]) + " 0 0 0 0 0 0 0 0\n")
			yolof.close()


	# Copy images from yolo to kitti
	copy_images_kitti(yolo_dir, kitti_dir)

###########################################################
##########        YOLO to LISA Conversion        ##########
###########################################################
def lisa(yolo_dir, lisa_dir, label=None):
	pass

###########################################################
##########        YOLO to VOC Conversion         ##########
###########################################################
def voc(yolo_dir, lisa_dir, label=None):
	pass