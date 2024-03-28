# PlaneBrain
2D representation of 3D neuroimaging data in form of mosaics with up to two segmentation results included

# Summary
Visualize MRI data with optional lesion and brain segmentations. Useful tool for quick and easy quality control of image sequences and segentation results. 

I recommend using the following after the image is created to 
1) convert it from png to jpg for space saving
2) trim down the white space

convert -fuzz 15% -trim {out file}.png {outfile}.jpg; rm {out file}.png;

# Description

This script loads MRI data from NIfTI files, optionally along with lesion and brain segmentations, and visualizes slices of these images. 

Users can specify slices to view and adjust image quality. 

The script supports visualization of the raw brain image, (optional) with overlaid segmentation outlines if provided.

# Requires

Python 3, NumPy, Matplotlib, Nibabel, scikit-image, pydicom.

# TODO
- DICOM load is experimental. Needs to be improved
- Implement functionality for automatic slice selection based on ROI.