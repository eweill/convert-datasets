# Import necessary libraries
import os, sys, shutil, glob, argparse
import numpy as np
from PIL import Image

# Import datasets dependent files
from datasets import kitti
from datasets import lisa
from datasets import voc
from datasets import yolo

def parse_args():
	"""
	Definition: Parse command line arguments.

	Parameters: None
	Returns: args - list of arguments
	"""
	parser = argparse.ArgumentParser(description=
		'Convert object detection datasets.')
	parser._action_groups.pop()
	required = parser.add_argument_group('required arguments')
	optional = parser.add_argument_group('optional arguments')
	required.add_argument('--from',
						  dest='from_key',
						  required=True,
						  help=f'Format to convert dataset from',
						  choices=['kitti','lisa','voc','yolo'],
						  type=str, nargs=1)
	required.add_argument('--from-path',
						  dest='from_path',
						  required=True,
						  help=f'Path to dataset you wish to convert.',
						  type=str, nargs=1)
	required.add_argument('--to',
                          dest='to_key',
                          required=True,
                          help=f'Format to convert dataset to',
                          choices=['kitti','lisa','voc','yolo'],
                          type=str, nargs=1)
	required.add_argument('--to-path',
						  dest='to_path',
						  required=True,
						  help=f'Path to output dataset to convert to.',
						  type=str, nargs=1)
	optional.add_argument('-l', '--label',
    					  dest='label',
    					  required=False,
    					  help=f'Label file necessary for yolo conversion.',
    					  type=str, nargs=1)
	optional.add_argument('-v','--verbose',
                          dest='verbose',
                          required=False,
                          help=f'Print out during execution of the script.')

	args = parser.parse_args()
	return args

if __name__ == '__main__':
	# Parse command line arguments
	args = parse_args()

	# If conversion types are same, no conversion necessary (ex. both 'kitti')
	if args.from_key == args.to_key:
		print ("No conversion necessary.")
		exit(0)

	# If yolo is part of the conversion (either 'to' or 'from' type)
	if 'yolo' in args.to_key or 'yolo' in args.from_key:
		# Must contain a label file
		if not args.label:
			print ("Error: A label file is necessary for yolo conversion.")
			exit(0)

		# Parameters including the label file
		params = ("'" + args.from_path[0] + "', '" + args.to_path[0] + "', '" +
			args.label[0] + "'")

	# Otherwise set up parameters without a label file
	else:
		# Parameters without the label file
		params = ("'" + args.from_path[0] + "', '" + args.to_path[0] + "'")

	# Evaluate the conversion based on command line parameters
	eval (args.from_key[0] + '.' + args.to_key[0] + '(' + params + ')')

	print ("Conversion complete!!")
