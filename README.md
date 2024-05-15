
# PlaneBrain
2D representation of 3D neuroimaging data in the form of mosaics with up to two segmentation results included

# Summary
Visualize MRI data with optional lesion and brain segmentations. Useful tool for quick and easy quality control of image sequences and segmentation results.

# Description
This script loads MRI data from NIfTI files, optionally along with lesion and brain segmentations, and visualizes slices of these images. Users can specify slices to view and adjust image quality. The script supports visualization of the raw brain image, optionally with overlaid segmentation outlines if provided. If no step size is specified, the script dynamically adapts to show a maximum of 50 equally spaced slices. Completely white rows and columns are removed from the final image.

# Requires
Python 3, NumPy, Matplotlib, Nibabel, scikit-image, pydicom, Pillow.

# Usage Example
```
python planebrain.py -i input_file.nii --l lesion_segmentation.nii --b brain_segmentation.nii --o output_file.png --s 2 --smin 10 --smax 50 --dpi 100
```

# TODO
- DICOM load is experimental. Needs to be improved.
- Implement functionality for automatic slice selection based on ROI.
- Make it work with 4D images
