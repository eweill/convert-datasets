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
from lxml import etree

python_version = sys.version_info.major

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
		if python_version == 3:
			prompt = input('Directory already exists. Overwrite? (yes, no): ')
		else:
			prompt = raw_input('Directory already exists. Overwrite? (yes, no): ')
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
			kittif = open(kitti_dir + "train/labels/" + f, "a+")
			for l, c in zip(labels, coords):
				kittif.write(l + " 0 0 0 " + str(c[0]) + " " + str(c[1]) +
					" " + str(c[2]) + " " + str(c[3]) + " 0 0 0 0 0 0 0 0\n")
			kittif.close()

	# Iterate through yolo validation data
	for f in os.listdir(yolo_dir + "val/labels/"):
		fname = (yolo_dir + "val/images/" + f).split(".txt")[0] + ".jpg"
		if os.path.isfile(fname):
			img = Image.open(fname)
			w, h = img.size
			img.close()
			labels, coords = parse_labels_kitti(os.path.join(yolo_dir + 
				"val/labels/" + f), labels_split, w, h)
			kittif = open(kitti_dir + "val/labels/" + f, "a+")
			for l, c in zip(labels, coords):
				kittif.write(l + " 0 0 0 " + str(c[0]) + " " + str(c[1]) +
					" " + str(c[2]) + " " + str(c[3]) + " 0 0 0 0 0 0 0 0\n")
			kittif.close()


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
def write_voc_file(fname, labels, coords, img_width, img_height):
	"""
	Definition: Writes label into VOC (XML) format.

	Parameters: fname - full file path to label file
				labels - list of objects in file
				coords - list of position of objects in file
				img_width - width of image
				img_height - height of image
	Returns: annotation - XML tree for image file
	"""
	annotation = etree.Element('annotation')
	filename = etree.Element('filename')
	f = fname.split("/")
	filename.text = f[-1]
	annotation.append(filename)
	folder = etree.Element('folder')
	folder.text = "/".join(f[:-1])
	annotation.append(folder)
	for i in range(len(coords)):
		object = etree.Element('object')
		annotation.append(object)
		name = etree.Element('name')
		name.text = labels[i]
		object.append(name)
		bndbox = etree.Element('bndbox')
		object.append(bndbox)
		xmax = etree.Element('xmax')
		xmax.text = str(coords[i][2])
		bndbox.append(xmax)
		xmin = etree.Element('xmin')
		xmin.text = str(coords[i][0])
		bndbox.append(xmin)
		ymax = etree.Element('ymax')
		ymax.text = str(coords[i][3])
		bndbox.append(ymax)
		ymin = etree.Element('ymin')
		ymin.text = str(coords[i][1])
		bndbox.append(ymin)
		difficult = etree.Element('difficult')
		difficult.text = '0'
		object.append(difficult)
		occluded = etree.Element('occluded')
		occluded.text = '0'
		object.append(occluded)
		pose = etree.Element('pose')
		pose.text = 'Unspecified'
		object.append(pose)
		truncated = etree.Element('truncated')
		truncated.text = '1'
		object.append(truncated)
	img_size = etree.Element('size')
	annotation.append(img_size)
	depth = etree.Element('depth')
	depth.text = '3'
	img_size.append(depth)
	height = etree.Element('height')
	height.text = str(img_height)
	img_size.append(height)
	width = etree.Element('width')
	width.text = str(img_width)
	img_size.append(width)

	return annotation

def determine_label_voc(label, labels):
	"""
	Definition: Converts label index to label from label set

	Parameters: label - label from file
				labels - list of labels
	Returns: label corresponding to index in yolo label
	"""
	return str(labels[int(label)])


def parse_labels_voc(label_file, labels, img_width, img_height):
	"""
	Definition: Parses label file to extract label and bounding box
		coordintates.

	Parameters: label_file - list of labels in images
	Returns: all_labels - contains a list of labels for objects in the image
			 all_coords - contains a list of coordinates for objects in image
	"""
	lfile = open(label_file)
	coords = []
	all_coords = []
	all_labels = []
	for line in lfile:
		l = line.split(" ")
		all_labels.append(determine_label_voc(l[0], labels))
		coords = list(map(float, list(map(float, l[1:5]))))
		xmin = float(img_width) * (2.0 * float(coords[0]) - float(coords[2])) / 2.0
		ymin = float(img_height) * (2.0 * float(coords[1]) - float(coords[3])) / 2.0
		xmax = float(img_width) * (2.0 * float(coords[0]) + float(coords[2])) / 2.0
		ymax = float(img_height) * (2.0 * float(coords[1]) + float(coords[3])) / 2.0
		tmp = [xmin, ymin, xmax, ymax]
		all_coords.append(list(map(int, tmp)))
	lfile.close()
	return all_labels, all_coords

def copy_images_voc(yolo, voc):
	"""
	Definition: Copy all images from the training and validation sets
		in kitti format to training and validation image sets in voc
		format.

	Parameters: yolo - path to yolo directory (contains 'train' and 'val')
				voc - path to voc output directory
	Returns: None
	"""
	for filename in glob.glob(os.path.join(yolo + "train/images/", "*.*")):
		shutil.copy(filename, voc + "train/images/")
	for filename in glob.glob(os.path.join(yolo + "val/images/", "*.*")):
		shutil.copy(filename, voc + "val/images/")

	for filename in glob.glob(os.path.join(voc + "train/images/", "*.*")):
		im = Image.open(filename)
		im.save(filename.split(".jpg")[0] + ".png", "png")
		os.remove(filename)
	for filename in glob.glob(os.path.join(voc + "val/images/", "*.*")):
		im = Image.open(filename)
		im.save(filename.split(".jpg")[0] + ".png", "png")
		os.remove(filename)

def make_voc_directories(voc):
	"""
	Definition: Make directories for voc images and labels.
		Removes previously created voc image and label directories.

	Parameters: yolo - path to voc directory to be created
	Returns: None
	"""
	if os.path.exists(voc):
		if python_version == 3:
			prompt = input('Directory already exists. Overwrite? (yes, no): ')
		else:
			prompt = raw_input('Directory already exists. Overwrite? (yes, no): ')
		if prompt == 'no':
			exit(0)
		shutil.rmtree(voc)
	os.makedirs(voc)
	os.makedirs(voc + "train")
	os.makedirs(voc + "train/images")
	os.makedirs(voc + "train/labels")
	os.makedirs(voc + "val")
	os.makedirs(voc + "val/images")
	os.makedirs(voc + "val/labels")

def voc(yolo_dir, voc_dir, label=None):
	print ("Convert yolo to voc")

	# Split label file
	label_file = open(label)
	labels_split = label_file.read().split('\n')

	# Make all directories for voc dataset
	make_voc_directories(voc_dir)

	# Iterate through kitti training data
	for f in os.listdir(yolo_dir + "train/labels/"):
		fname = (yolo_dir + "train/images/" + f).split(".txt")[0] + ".jpg"
		if os.path.isfile(fname):
			img = Image.open(fname)
			w, h = img.size
			img.close()
			labels, coords = parse_labels_voc(os.path.join(yolo_dir +
				"train/labels/" + f), labels_split, w, h)
			annotation = write_voc_file(fname, labels, coords, w, h)
			et = etree.ElementTree(annotation)
			et.write(voc_dir + "train/labels/" + f.split(".txt")[0] + ".xml", pretty_print=True)

	# Iterate through kitti validation data
	for f in os.listdir(yolo_dir + "val/labels/"):
		fname = (yolo_dir + "val/images/" + f).split(".txt")[0] + ".jpg"
		if os.path.isfile(fname):
			img = Image.open(fname)
			w, h = img.size
			img.close()
			labels, coords = parse_labels_voc(os.path.join(yolo_dir +
				"val/labels/" + f), labels_split, w, h)
			annotation = write_voc_file(fname, labels, coords, w, h)
			et = etree.ElementTree(annotation)
			et.write(voc_dir + "val/labels/" + f.split(".txt")[0] + ".xml", pretty_print=True)

	# Copy images from kitti to voc
	copy_images_voc(yolo_dir, voc_dir)

