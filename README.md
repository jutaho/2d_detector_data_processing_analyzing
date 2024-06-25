# Data Processing & Analysis
	### Test
## Acquisition: XISL to controll XRD 0822
	### Set gain and integration time
	### Choose directory to store measurement data
	### Start and stop measurement data
	### Save measurement data

## Processing XRD 0822 Data .his
	### Read header info: Number of pixels per dimension, pixel distance, integration time, gain setting
	### Read raw data frames
	### Specify frames in terms of exposure
	### Perform background correction
	### Generate bad pixel matrix
	### Estimate spatial and temporal noise
	### Perform bad pixel correction
	### Perform uniformity correction
	### Apply 2D median filter
	### Write processed data 
	### Write acquisition information = Integration time, gain, number of frames, %FSR, noise,
	### Write correction related information = Bad pixel & sensitivity matrices, background image, median filter kernel size

## Processing Octavius 1600 XDR Data .mcc
	### Read raw data
	### Interpolate NaNs
	### Interpolate to 0.1 mm grid 
	### Write tiff file 16 bit 

## Data Analysis
	### Perform field flatness analysis
	### Perform spot size and position analysis
	### Perform spatial frequency analysis
	### Compare data
	### Check image quality 

