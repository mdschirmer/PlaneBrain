
# PlaneBrain
2D representation of 3D neuroimaging data in the form of mosaics with up to two segmentation results included

# Summary
Visualize MRI data with optional lesion and brain segmentations. Useful tool for quick and easy quality control of image sequences and segmentation results.

# Description
This script loads MRI data from NIfTI files, optionally along with lesion and brain segmentations, and visualizes slices of these images. Users can specify slices to view and adjust image quality. The script supports visualization of the raw brain image, optionally with overlaid segmentation outlines if provided. If no step size is specified, the script dynamically adapts to show a maximum of 50 equally spaced slices. Completely white rows and columns are removed from the final image. The script also handles 4D images by creating a mosaic for each 3D volume within the 4D file and saving the output with a modified file name that includes the volume number.

# Requires
Python 3, NumPy, Matplotlib, Nibabel, scikit-image, pydicom, Pillow.

# Usage Example
```
python planebrain.py -i input_file.nii.gz --l lesion_segmentation.nii.gz --b brain_segmentation.nii.gz --o output_file.jpg --s 2 --smin 10 --smax 50 --dpi 100
```

# TODO
- DICOM load is experimental. Needs to be improved.
- Implement functionality for automatic slice selection based on ROI.
