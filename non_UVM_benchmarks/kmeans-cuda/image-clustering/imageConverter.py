
#Import modules
import sys, os, math, time
import numpy as np
from PIL import Image

#Global Variables
#Width
W = 0
#Height
H = 0
# File Name without extension
global NAMEFILE
#Current file extension
global EXTENSION

#Used to specify raw file extension
global RAW


def fromJpgToUint8():
	global W,H
	img = Image.open(NAMEFILE+EXTENSION)
	W = img.size[0]
	H = img.size[1]
	print "Width: " + str(W) + " Height: " + str(H)

	#print type(img)
	array = np.asarray(img)
	#print type(array)
	#print array.dtype
	
	# Debugging..
	# (300,400,3) means 400x300 with RGB channels
	#print "Array shape: " + str(array.shape)
	# Starting from top left bottom
	
	#print "Array: "
	#print array
	# 'uint8'
	#print "Array Type: "+ str(array.dtype) 
	# Saving the array in .raw format
	print "Saving .raw ..."
	array.tofile(NAMEFILE+RAW)
	print "Saved!"
	

def fromUint8ToJpg():
	# Reading .raw file
	array = np.fromfile(NAMEFILE+EXTENSION, dtype = "uint8")
	#print array
	#print len(array)
	# Reshaping 1d Array into 2d Array
	outputImage = array.reshape(H,W,3)
	image = Image.fromarray(outputImage)
	print "Saving .jpg ...."
	# Saving new image
	image.save(NAMEFILE+"_PYoutput"+".jpg" ,quality=75)
	print "Saved!"
	

if __name__ == "__main__":
	global NAMEFILE, EXTENSION, RAW, W, H
	RAW = ".raw"
	args = sys.argv[1:]
	inputFile = args[0]
	print "File Name: " + str(inputFile)
	try:
		NAMEFILE,EXTENSION = os.path.splitext(inputFile)
		if  (EXTENSION.lower() == ".jpg" or EXTENSION.lower() == ".jpeg"):
			fromJpgToUint8()
		elif EXTENSION.lower() == RAW:
			W = int(args[1])
			H = int(args[2])
			fromUint8ToJpg()
		else:
			print "Error: Unknown image format."
	except IOError:
		print "Unable to load your image."

