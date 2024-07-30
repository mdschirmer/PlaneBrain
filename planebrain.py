#!/usr/bin/env python3

"""
PlaneBrain - 2D representation of 3D neuroimaging data in the form of mosaics with up to two segmentation results included

:Summary: Visualize MRI data with optional lesion and brain segmentations. Useful tool for quick and easy quality control of image sequences and segmentation results.

:Description: This script loads MRI data from NIfTI files, optionally along with lesion and brain segmentations, and visualizes slices of these images. 
Users can specify slices to view and adjust image quality. 
The script supports visualization of the raw brain image, optionally with overlaid segmentation outlines if provided. 
If no step size is specified, the script dynamically adapts to show a maximum of 50 equally spaced slices. 
Completely white rows and columns are removed from the final image. The script also handles 4D images by creating a mosaic for each 3D volume within the 4D file and saving the output with a modified file name that includes the volume number.

:Requires: Python 3, NumPy, Matplotlib, Nibabel, scikit-image, pydicom, Pillow.

:Usage Example:
    python plane_brain.py -i input_file.nii --l lesion_segmentation.nii.gz --b brain_segmentation.nii.gz --o output_file.jpg --s 2 --smin 10 --smax 50 --dpi 100

:Parameters:
    -i <input_file> : Input NIfTI file or directory containing DICOM files.
    --l <lesion_file> : (Optional) Lesion segmentation NIfTI file.
    --b <brain_file> : (Optional) Brain segmentation NIfTI file.
    --o <output_file> : Output image file name (default: file.png).
    --s <slice_step> : (Optional) Show only every nth slice (default: 2).
    --g <guide> : (Optional) Draw border around slices containing lesion overlays.
    --smin <min_slice> : (Optional) Minimum slice number.
    --smax <max_slice> : (Optional) Maximum slice number.
    --dpi <dpi> : (Optional) Set output image dpi (default: 80).
    --dcm : (Optional) Set input file format to be DICOM: -i option needs to be a folder.

:TODO: 
- Implement functionality for automatic slice selection based on ROI.
- Ensure the script logs the command used and the error to failed.log if it fails.
"""

#=============================================
# Metadata
#=============================================
__author__ = 'MDS, Refactored by LAA'
__organization__ = ''
__contact__ = ''
__copyright__ = 'CC-BY'
__license__ = 'http://www.opensource.org/licenses/mit-license.php'
__date__ = '2024-05-21'
__version__ = '0.5'

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
from PIL import Image, ImageChops, ImageOps
import re

import pdb

#=============================================
# Helper functions
#=============================================

def remove_white_rows_cols_with_size(img_array, original_width, original_height):
    # Identify rows and columns that are not completely white
    non_white_rows = np.where(~np.all(img_array == 255, axis=(1, 2)))[0]
    non_white_cols = np.where(~np.all(img_array == 255, axis=(0, 2)))[0]
    
    # Slice the array to keep only non-white rows and columns
    trimmed_array = img_array[non_white_rows[:, None], non_white_cols, :]
    
    # Convert to image and resize to maintain original size
    img = Image.fromarray(trimmed_array)
    img = img.resize((original_width, original_height), Image.LANCZOS)
    
    return np.array(img)

def get_slice(img, n):
    # Select the nth slice of the image, handling out-of-range values
    if n >= img.shape[2]:
        n = img.shape[2] - 1
    sli = img[:, :, n]

    # Rotate the slice for proper orientation
    return np.rot90(sli, 1)

def get_outline(img, n):
    # Similar to get_slice but also finds contours at a specified threshold
    if n >= img.shape[2]:
        n = img.shape[2] - 1
    sli = img[:, :, n]
    sli = np.rot90(sli, 1)

    contours = skimage.measure.find_contours(sli, 0.8)

    return contours

def cut_img(img, smin, smax):
    # Cut the image to include only slices within a specified range
    if smax is not None:
        img = img[:, :, :int(smax)]

    if smin is not None:
        img = img[:, :, int(smin):]

    return img

def read_dicom_folder(dcmpath):
    lstFilesDCM = []  # create an empty list
    for dirName, subdirList, fileList in os.walk(dcmpath):
        for filename in fileList:
            lstFilesDCM.append(os.path.join(dirName, filename))
                
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

def draw_frame(rows, cols, fill_value):
    border_width = 4
    canvas = np.zeros((rows, cols)) #nans are ignored in plt.imshow() allowing for colored frame with transparent center 
    canvas[:, 0:int(border_width/2)] = fill_value
    canvas[:, -int(border_width/2):] = fill_value
    canvas[0:border_width, :] = fill_value
    canvas[-border_width:, :] = fill_value
    return canvas

def generate_bounds(stacked, row_idx, col_idx, slice_row_size, slice_col_size):
    if stacked == True:
        slice_0_r_lb = int(2*row_idx*slice_row_size)
        slice_0_r_ub = int((row_idx*2+1)*slice_row_size) #first row upper bound, second row lower bound
        slice_1_r_ub = int((row_idx*2+2)*slice_row_size) #second row upper bound
        slice_c_lb = int(col_idx*slice_col_size) #column bounds only need to be calculated once
        slice_c_ub = int((col_idx+1)*slice_col_size)
        return [slice_0_r_lb, slice_0_r_ub, slice_1_r_ub, slice_c_lb, slice_c_ub]
    else:
        slice_0_r_lb = int(row_idx*slice_row_size)
        slice_0_r_ub = int((row_idx+1)*slice_row_size) #first row upper bound
        slice_1_r_ub = 0 #place holder
        slice_c_lb = int(col_idx*slice_col_size) #column bounds only need to be calculated once
        slice_c_ub = int((col_idx+1)*slice_col_size)
        return [slice_0_r_lb, slice_0_r_ub, slice_1_r_ub, slice_c_lb, slice_c_ub]

#=============================================
# Main method
#=============================================
import traceback

def main(argv):
    try:
        # Load image
        if argv.dcm is False:
            nii = nib.load(argv.i)
            img = nii.get_fdata()
        else:
            assert os.path.isdir(argv.i), "DCM flag specified, input should be a folder"
            img = np.rot90(read_dicom_folder(argv.i), -1)

        img = cut_img(img, argv.smin, argv.smax)

        # Check if the image is 4D
        if img.ndim == 4:
            volumes = img.shape[3]
        else:
            volumes = 1

        for vol in range(volumes):
            if volumes > 1:
                img_vol = img[:, :, :, vol]
                output_file = re.sub(r'(\.\w+)$', f'_vol{vol:04d}\\1', argv.o)
            else:
                img_vol = img
                output_file = argv.o

            # Load segmentations if specified
            seg, brain = None, None
            if argv.l is not None:
                nii = nib.load(argv.l)
                seg = nii.get_fdata()
                seg = cut_img(seg, argv.smin, argv.smax)
                if volumes > 1:
                    seg = seg[:, :, :, vol]

            if argv.b is not None:
                nii = nib.load(argv.b)
                brain = nii.get_fdata()
                brain = cut_img(brain, argv.smin, argv.smax)
                if volumes > 1:
                    brain = brain[:, :, :, vol]

            # Define plot parameters
            dpi = float(argv.dpi)
            if argv.s is not None:
                step_size = int(argv.s)
                img_slices = np.arange(0, img_vol.shape[2], step_size).astype(int)
            else:
                step_size = max(1, img_vol.shape[2] // 50)  # Ensure at most 50 slices
                img_slices = np.arange(0, img_vol.shape[2], step_size).astype(int)
            num_slices = len(img_slices)

            # Define maximum number of images per row
            max_img_per_row = 10

            # Calculate the number of images per row dynamically
            img_per_row = min(max_img_per_row, num_slices)

            # Calculate the number of rows needed
            if seg is not None or brain is not None:
                rows_needed = (num_slices + img_per_row - 1) // img_per_row * 2  # Double the number of rows if segmentation overlays are present
                stacked = True
            else:
                rows_needed = (num_slices + img_per_row - 1) // img_per_row
                stacked = False
            
            # Render figure as an array based on number of rows needed and images per row. Assumes all slices have same row x column dimensions
            slice_row_size, slice_col_size = get_slice(img_vol, img_slices[0]).shape # Gets pixel dimensions of each slice
            figure_array = np.zeros((3, slice_row_size*rows_needed, slice_col_size*img_per_row)) #Makes an array large enough to hold all frames
            frame = draw_frame(slice_row_size, slice_col_size, 255) #renders frame once
            l_w = 1 # Lineweight modifier for outlines. Final lineweight in pix is 1+2*l_w 

            for ii, img_slice in enumerate(img_slices):
                # Generate upper and lower bounds for each slice's position in the array:
                row_idx = ii // img_per_row
                col_idx = ii % img_per_row
                bounds = generate_bounds(stacked= stacked, row_idx=row_idx, col_idx=col_idx, slice_row_size= slice_row_size, slice_col_size= slice_col_size)
                
                # Insert base slices into first layer 
                base_slice = get_slice(img_vol, img_slice) 
                figure_array[0][bounds[0]:bounds[1], bounds[3]:bounds[4]] = base_slice 
                
                 
                # Insert second row of slices if applicable
                if seg is not None or brain is not None: 
                    figure_array[0][bounds[1]:bounds[2], bounds[3]:bounds[4]] = base_slice
                
                # Generate lesion mask and add border to second layer
                if seg is not None: 
                    contours = get_outline(seg, img_slice)
                    lesion_mask = np.zeros((base_slice.shape))
                    for c in contours:
                        for p in c: 
                            lesion_mask[int(p[0]-l_w):int(p[0]+l_w), int(p[1]-l_w):int(p[1]+l_w)] = 1 
                        if argv.g is True:
                            figure_array[1][bounds[1]:bounds[2], bounds[3]:bounds[4]] = lesion_mask + frame 
                        else:
                            figure_array[1][bounds[1]:bounds[2], bounds[3]:bounds[4]] = lesion_mask

                # Generate brain mask and add to third layer.
                if brain is not None: 
                    contours = get_outline(brain, img_slice)
                    brain_mask = np.zeros((base_slice.shape))
                    for c in contours:
                        for p in c:
                            brain_mask[int(p[0]-l_w):int(p[0]+l_w), int(p[1]-l_w):int(p[1]+l_w)]  = 255
                    figure_array[2][bounds[1]:bounds[2], bounds[3]:bounds[4]] = brain_mask 

            # Generate masked arrays to display figure properly
            masked_lesion = np.ma.masked_where(figure_array[1] == 0, figure_array[1]) 
            masked_brain = np.ma.masked_where(figure_array[2] == 0, figure_array[2])

            # Display base layer with brain scans
            plt.imshow(figure_array[0], cmap= 'gray') 
            
            # Display lesion or brain overlays if available:
            if seg is not None:
                plt.imshow(masked_lesion, cmap= 'autumn', interpolation='none') # In autumn cmap, low values are red and high are yellow. 

            if brain is not None:
                plt.imshow(masked_brain, cmap= 'cool', interpolation='none')

            #Output figure
            plt.axis('off')
            plt.savefig(output_file, dpi = 1000, pad_inches = 0, bbox_inches = 'tight') 
            

                
            """    
            # Adjust figure size to minimize white space and ensure gsizeood visibility
            fig_width = img_per_row * 3  # Increase width to reduce thin columns
            fig_height = fig_width * (rows_needed / img_per_row)  # Calculate height based on the ratio of total slices to images per row
            fig = plt.figure(figsize=(fig_width, fig_height))

            # Create axes for plotting and remove white space/axis where possible
            ax = gridspec.GridSpec(rows_needed, img_per_row, wspace=0, hspace=0)

            
            for ii in range(rows_needed * img_per_row):
                a = plt.subplot(ax[ii])
                a.axis('off')

            for ii, img_slice in enumerate(img_slices):
                row_idx = ii // img_per_row
                col_idx = ii % img_per_row

                if seg is not None or brain is not None:
                    # Plot brain slices in odd rows
                    a = plt.subplot(ax[row_idx * 2, col_idx])
                    a.imshow(get_slice(img_vol, img_slice), cmap='gray')
                    a.axis('off')

                    # Plot segmentation overlays in even rows if available
                    overlay_row_idx = row_idx * 2 + 1
                    a = plt.subplot(ax[overlay_row_idx, col_idx])
                    segmentation_slice = get_slice(img_vol, img_slice)
                    slice_rows, slice_cols = segmentation_slice.shape
                    a.imshow(segmentation_slice, cmap= 'gray')
                    #a.imshow(get_slice(img_vol, img_slice), cmap='gray')
                    a.axis('off')

                    contrast_multpilier = 50 #Multiplies contours to increase brightnes. 50 is arbitrary. 
                    if brain is not None:
                        contours = get_outline(brain, img_slice) * contrast_multpilier
                        for contour in contours:
                            a.plot(contour[:, 1], contour[:, 0], linewidth=.65, color='aqua') 
         
                    if seg is not None:
                        contours = get_outline(seg, img_slice) * contrast_multpilier
                        for contour in contours:
                            a.plot(contour[:, 1], contour[:, 0], linewidth=.65, color='red')
                            if argv.g is True:
                                frame = draw_frame(slice_rows, slice_cols, np.max(segmentation_slice))
                                a.imshow(frame, cmap= "autumn")
                else:
                    # Plot brain slices when no segmentations are available
                    a = plt.subplot(ax[row_idx, col_idx])
                    a.imshow(get_slice(img_vol, img_slice), cmap='gray')
                    a.axis('off')

            # Convert figure to array, remove white rows and columns while maintaining size
            fig.canvas.draw()
            img_data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img_data = img_data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            original_width, original_height = fig.canvas.get_width_height()

            # Remove white rows and columns and maintain size
            trimmed_data = remove_white_rows_cols_with_size(img_data, original_width, original_height)

            # Convert trimmed array back to image and save
            trimmed_img = Image.fromarray(trimmed_data)
            trimmed_img.save(output_file)
            """
    except Exception as e:
        # Write the command that was used to call the script to failed.log
        output_dir = os.path.dirname(argv.o)
        with open(os.path.join(output_dir, 'failed.log'), 'a') as log_file:
            log_file.write(f"Command: {' '.join(sys.argv)}\n")
            log_file.write(f"===============================\n")
            log_file.write(f"Error: {str(e)}\n")
            log_file.write(traceback.format_exc())
            log_file.write(f"#############################\n")


if __name__ == "__main__":
    # Catch input
    try:
        parser = OptionParser()
        parser.add_option('-i', dest='i', help='Input FILE', metavar='FILE')
        parser.add_option('--l', dest='l', help='Lesion segmentation FILE', metavar='FILE', default=None)
        parser.add_option('--b', dest='b', help='Brain segmentation FILE', metavar='FILE', default=None)
        parser.add_option('--o', dest='o', help='Output FILE', metavar='FILE', default='file.png')
        parser.add_option('--s', dest='s', help='Show only every s slice (default = 2)', metavar='INTEGER', default=None)
        parser.add_option('--g', action='store_true', dest='g', help='Display frame around image slices containing lesion segmentations.', metavar = 'BOOL', default=False)
        parser.add_option('--smin', dest='smin', help='Minimum slice number', metavar='INTEGER', default=None)
        parser.add_option('--smax', dest='smax', help='Maximum slice number', metavar='INTEGER', default=None)
        parser.add_option('--dpi', dest='dpi', help='Set output image dpi', metavar='INTEGER', default=80)
        parser.add_option('--dcm', action="store_true", dest='dcm', help='Set input file format to be dcm: -i option needs to be a folder', metavar='BOOL', default=False)
        (options, args) = parser.parse_args()
    except:
        sys.exit()

    main(options)
