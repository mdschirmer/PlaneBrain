#!/usr/bin/env python3
"""
PlaneBrain - 2D representation of 3D neuroimaging data in form of mosaics with up to two segmentation results included

:Summary: Visualize MRI data with optional lesion and brain segmentations. Useful tool for quick and easy quality control of image sequences and segentation results. 

I recommend using the following after the image is created to 
1) convert it from png to jpg for space saving
2) trim down the white space

convert -fuzz 15% -trim {out file}.png {outfile}.jpg; rm {out file}.png;

:Description: This script loads MRI data from NIfTI files, optionally along with lesion and brain segmentations, and visualizes slices of these images. 
Users can specify slices to view and adjust image quality. 
The script supports visualization of the raw brain image, (optional) with overlaid segmentation outlines if provided.

:Requires: Python 3, NumPy, Matplotlib, Nibabel, scikit-image, pydicom.

:TODO: 
- Implement functionality for automatic slice selection based on ROI.
"""

#=============================================
# Metadata
#=============================================
__author__ = 'MDS'
__organization__ = 'HMS/MGB'
__contact__ = 'mdschirmer @ github'
__copyright__ = 'CC-BY'
__license__ = 'http://www.opensource.org/licenses/mit-license.php'
__date__ = '2023-03'
__version__ = '0.2'

#=============================================
# Import statements
#=============================================
import sys
import os
from optparse import OptionParser
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import skimage
import matplotlib.gridspec as gridspec
import pydicom

import pdb
#=============================================
# Helper functions
#=============================================

def get_slice(img, n):
	# Select the nth slice of the image, with somewhat handling out-of-range values
	#assert ((n>=0) & (n<(img.shape[2]))), "slice number outside of possible range"
	if n>=img.shape[2]:
		n=img.shape[2]-1
	sli = img[:,:,n]

	# Rotate the slice for proper orientation
	return np.rot90(sli,1)

def get_outline(img, n):
	# Similar to get_slice but also finds contours at a specified threshold
	#assert ((n>=0) & (n<(img.shape[2]))), "slice number outside of possible range"
	if n>=img.shape[2]:
		n=img.shape[2]-1
	sli = img[:,:,n]
	sli = np.rot90(sli,1)

	contours = skimage.measure.find_contours(sli, 0.8)

	return contours

def cut_img(img, smin, smax):
	# Cut the image to include only slices within a specified range
	if smax is not None:
		img = img[:,:,:int(smax)]

	if smin is not None:
		img = img[:,:,int(smin):]

	return img

def read_dicom_folder(dcmpath):
	lstFilesDCM = []  # create an empty list
	for dirName, subdirList, fileList in os.walk(dcmpath):
		for filename in fileList:
			lstFilesDCM.append(os.path.join(dirName,filename))
				
	# Get ref file
	RefDs = pydicom.read_file(lstFilesDCM[0])

	# Load dimensions based on the number of rows, columns, and slices (along the Z axis)
	ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns), len(lstFilesDCM))

	# Load spacing values (in mm)
	ConstPixelSpacing = (float(RefDs.PixelSpacing[0]), float(RefDs.PixelSpacing[1]), float(RefDs.SliceThickness))

	# The array is sized based on 'ConstPixelDims'
	ArrayDicom = np.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)

	# get slice locations and then sort
	locations = []
	for filenameDCM in lstFilesDCM:
		# read the file
		ds = pydicom.read_file(filenameDCM)
		locations.append(float(ds.SliceLocation))

	# get the index
	idx = np.argsort(locations)

	# loop through all the DICOM files
	for ii, iFile in enumerate(idx):
		filenameDCM = lstFilesDCM[iFile]
		# read the file
		ds = pydicom.read_file(filenameDCM)
		# store the raw image data
		ArrayDicom[:, :, ii] = ds.pixel_array

	return ArrayDicom

#=============================================
# Main method
#=============================================
def main(argv):
	# load image
	if argv.dcm is False:
		nii = nib.load(argv.i)
		img = nii.get_fdata()
	else:
		assert (os.path.isdir(argv.i)), "DCM flag specified, "
		img = np.rot90(read_dicom_folder(argv.i),-1)

	img = cut_img(img,argv.smin, argv.smax)

	# if segmentation file is specified, load segmentation
	if argv.l is not None:
		nii = nib.load(argv.l)
		seg = nii.get_fdata()
		seg = cut_img(seg,argv.smin, argv.smax)

	# load second segmentation image, e.g. brain seg
	if argv.b is not None:
		nii = nib.load(argv.b)
		brain = nii.get_fdata()
		brain = cut_img(brain,argv.smin, argv.smax)

	# define plot parameters
	img_per_row = 5
	img_per_clm = 10
	dpi = float(argv.dpi)
	fig = plt.figure(figsize=(6,12))
	
	# create axes for plotting and remove white space/axis where possible
	ax = gridspec.GridSpec(img_per_clm, img_per_row)
	ax.update(wspace=0, hspace=0)
	for ii in range(img_per_clm* img_per_row):
		a = plt.subplot(ax[ii])
		a.axis('off')

	# plot the raw brain image
	step_size = int(argv.s)
	for ii, img_slice in enumerate(np.arange(img.shape[2], step=step_size).astype(int)[:(img_per_row*np.floor((img_per_clm/2)).astype(int))]):
		a = plt.subplot(ax[2*(ii//img_per_row),(ii%img_per_row)])
		a.imshow(get_slice(img, img_slice), cmap='gray')
		a.axis('off')
		a.set_xticklabels([])
		a.set_yticklabels([])
		a.set_aspect('auto')

		# plot the segmentations
		if argv.l is not None or argv.b is not None:
			a = plt.subplot(ax[(ii//img_per_row)*2 + 1,(ii%img_per_row)])
			a.imshow(get_slice(img, img_slice), cmap='gray')
			a.axis('off')
			a.set_xticklabels([])
			a.set_yticklabels([])
			a.set_aspect('auto')

		# plot outline
		if argv.b is not None:
			contours = get_outline(brain, img_slice)
			for contour in contours:
				a.plot(contour[:, 1], contour[:, 0], linewidth=.5, color='orange')

		# plot lesions
		if argv.l is not None: 
			contours = get_outline(seg, img_slice)
			for contour in contours:
				a.plot(contour[:, 1], contour[:, 0], linewidth=.5, color='r')

	fig.savefig(argv.o,bbox_inches='tight', dpi=dpi)

	return 0

if __name__ == "__main__":
	# catch input
	try:
		parser = OptionParser()
		parser.add_option('-i', dest='i', help='Input FILE', metavar='FILE')
		parser.add_option('--l', dest='l', help='Lesion segmentation FILE', metavar='FILE',default=None)
		parser.add_option('--b', dest='b', help='Brain segmentation FILE', metavar='FILE',default=None)
		parser.add_option('--o', dest='o', help='Output FILE', metavar='FILE', default='file.png')
		parser.add_option('--s', dest='s', help='Show only every s slice (default = 2)', metavar='INTEGER', default=2)
		parser.add_option('--smin', dest='smin', help='Minimum slice number', metavar='INTEGER', default=None)
		parser.add_option('--smax', dest='smax', help='Maximum slice number', metavar='INTEGER', default=None)
		parser.add_option('--dpi', dest='dpi', help='Set output image dpi', metavar='INTEGER', default=80)
		parser.add_option('--dcm', action="store_true", dest='dcm', help='Set input file format to be dcm: -i option needs to be a folder', metavar='BOOL', default=False)
		(options, args) = parser.parse_args()
	except:
		sys.exit()

main(options)
