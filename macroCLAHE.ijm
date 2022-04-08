for (i = 1; i <= nSlices; i++) {
    setSlice(i);
    run("Enhance Local Contrast (CLAHE)","blocksize=10 histogram=256 maximum=3 mask=*None* fast_(less_accurate)");
}