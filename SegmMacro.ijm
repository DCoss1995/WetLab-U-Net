// Open file
run("Open...");

// Find edges and convert to mask
run("Find Edges","stack");
run("Convert to Mask", "method=Default background=Default calculate black");

// Add the mask as a ROI selection and save the ROIset
for (a=1; a<=nSlices; a++) {
	setSlice(a);
	run("Create Selection");
roiManager("add");
}
roiManager("deselect");
pathToROI = getDir("Where do you want to save the ROI set?");
roiManager("Save",pathToROI+"RoiSet.zip");

// Enlarge ROIs and save the new ROIset
b = roiManager("count");
for (c = 0; c<b; c++) {
	roiManager("select", c);
	run("Enlarge...","enlarge=8");
roiManager("Update");
}
pathToEnlROI = getDir("Where do you want to save the enlarged ROI set?");
roiManager("Save",pathToEnlROI+"EnlRoiSet.zip");

// Open the arteries file
run("Open...");

// Use the EnlRoiSet for arteries segmentation
d = roiManager("count");
for (e=0; e<d; e++) {
	roiManager("select",e);
	//run("Make Inverse");
	setBackgroundColor(0, 0, 0);
	run("Clear","slice");
}

// Verify segmentation
run("3D Project...", "projection=[Brightest Point] axis=Y-Axis slice=1 initial=0 total=360 rotation=10 lower=1 upper=255 opacity=0 surface=100 interior=50");
