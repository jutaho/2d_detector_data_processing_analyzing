# Data Processing & Analysis
	### Test
## Acquisition: XISL to controll XRD 0822
	### Set gain and integration time
	### Choose directory to store measurement data
	### Start and stop measurement data
	### Save measurement data

## Processing XRD 0822 Data
	### Read header info: Number of pixels per dimension, pixel distance, integration time, gain setting
	### Read raw data frames
	### Specify frames in terms of exposure
	### Perform background correction
	### Generate bad pixel map
	### Estimate spatial and temporal noise
	### Perform bad pixel correction
	### Perform pixel sensitivity correction
	### Apply 2D median filter
	### Write processed data 
	### Write acquisition information = Integration time, gain, number of frames, %FSR, noise,
	### Write correction related information = Bad pixel & sensitivity matrices, background image, median filter kernel size

## Processing of Octavius 1600 XDR Data
	### Read raw data
	### Estimate spatial and temporal noise
	### Write corrected data
	### Write acquisition information: %FSR, noise, SNR

## Data Analysis
	### Perform 2D flatness analysis
	### Perform spot size and position analysis
	### Perform spatial frequency analysis
	### Compare data
	### Check image quality 

