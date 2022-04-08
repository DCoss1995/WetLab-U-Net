# WetLab-U-Net
Quick introduction to what you will find in this repository:
- macroCLAHE.ijm is a macro for FIJI (https://imagej.net/) that allows to run the CLAHE (Contrast Limited Adaptive Histogram Equalization) on the whole stack.
- SegmMacro.ijm is a macro for FIJI that allows to use a mask to create a ROIset, and eventually enlarge it. Then it uses the ROIs in the ROIset to segment the image or the stack of your choice.
- Sort-And-Rename-Files.py is a Python script written for the creation of datasets. If you have an U-Net and the images on your platform, you can use this script to randomly sort your files into training and validation dataaset.
- Unet.py is a Python script with the U-Net alghoritm, with references. 

In some cases there aren't comments on what the line of code does, but I will add comments as soon as possible.
