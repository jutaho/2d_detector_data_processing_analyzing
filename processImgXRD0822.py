#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Description -     processes raw data from file ".his"-format of
#                   flatpanel XRD-0822 and writes DICOM file
# Author -          J. Horn

# Import modules
import datetime
import math
from tifffile import imwrite
import numpy as np
from scipy import signal
import scipy.ndimage

# Process raw frames
def processRawFrames(file, medFiltKernelSize):
    # Define variables
    date = datetime.datetime.now()
    formatLong = "%Y-%m-%d %H:%M:%S"
    formatShort  = "%Y%m%d"
    dateTime = date.strftime(formatLong)
    date = date.strftime(formatShort)
    folderFileName = file
    flagNoExp = 0
    pxlShiftInMm = 0.2
    bitResolution = 2**16 - 1
    violationLimitADC = bitResolution - 2
    tolAmountOfADCLimitViol = 25
    initNumOfBgFr = 5 
    integrationThresFactor = 6
    badPxlFactor = 3
    bg = 0
    
    # Load correction matrices
    # Bad pixel matrix will calculated based on background frames pre exposure
    pxlGainMatrix = np.load("pxlSensitivityMatrix.npy")
    
    ## Read raw frames
    # Open file, read header information
    # - Still not getting all header information: Integraton time, gain
    # - pixel distance 
    with open(folderFileName, mode="rb") as header: 
        headerInfo = np.fromfile(header, np.uint16, count = 16, offset = 0)
        numOfFrames = (headerInfo[10])   
        numOfPxlDim = (headerInfo[8:10])

    # Open file, skip header and read raw frames       
    with open(folderFileName, mode="rb") as rawData:    
        frames = np.fromfile(rawData, np.uint16,
        count = int(numOfFrames) * int(numOfPxlDim[0]) * int(numOfPxlDim[1]),
        offset = 100).reshape((numOfFrames, numOfPxlDim[0], numOfPxlDim[1]))
        
    print("\n#### #### #### #### #### #### #### #### #### #### #### #### ####")
    print(file)
    
    # Define pixel exceeding specified limits
    pxlViolLimitADCPerFrame = np.array(frames > violationLimitADC)
    countPxlOverViolLimit = [np.sum(pxlViolLimitADCPerFrame[frames]) 
                                for frames in range(len(frames))]

    ## Classify frames ##
    # Calculate spatial mean and standard deviation for each frame
    medSigVector = np.median(frames, axis = (1,2))
    stdSigVector = np.std(frames, axis = (1,2))
    
    # Handling "0" entrie frames that appear when aqcuisition stopped manually
    # before total number of frames is reached
    # Calculate valid number of frames
    indexStdSigEq0 = np.concatenate(np.where(medSigVector + stdSigVector == 0))
    if np.any(indexStdSigEq0) == False:
        lastFrameIndex = numOfFrames
    else:
        lastFrameIndex = indexStdSigEq0[0]
        
    print("Total number of frames: %i" %numOfFrames)
    print("Total number of valid frames: %i" %lastFrameIndex)

    # Classify beam on frame
    # Initial background correction for all frames
    initBgImg = np.mean(frames[0:initNumOfBgFr], axis = 0)
    initBgCorrImg = frames - initBgImg
    
    # Calculating threshold to check for exposure
    initExpThres = np.mean(
    initBgCorrImg[0:initNumOfBgFr]) + 6 * np.std(initBgCorrImg[0:initNumOfBgFr])
    
    # Create boolean arrays that comprise pixel over threshold == True
    exposureArrays = initBgCorrImg > initExpThres
    expCountPerFr = np.sum(exposureArrays, axis = (1,2))
    
    # Find Indices of exposed frames
    frIndicesBeamOn = np.concatenate(np.where(
    expCountPerFr > 1000)) # number needs to checked for robustness
    
    # Check frames for exposure
    if np.sum(frIndicesBeamOn) == 0:
        flagNoExp = 1
        print("No exposed frames detected...")
        background = np.mean(frames, axis = 0)
        backgroundTempNoise = np.std(frames, axis = 0)
        bg = [background, backgroundTempNoise]
        procImage = 0
        imgQual = 0
        imgs2Corr = 0
        return procImage, imgQual, pxlShiftInMm, imgs2Corr, bg  
    else:
        numOfExpFrames = len(frIndicesBeamOn)
   
    # Define background image pre and post exposure 
    frIndicesBgPre = range(frIndicesBeamOn[0])
    numOfBgFramesPre = len(frIndicesBgPre)
    frIndicesBgPost = range(frIndicesBeamOn[-1]+1,lastFrameIndex)
    numOfBgFramesPost = len(frIndicesBgPost)
    
    # If there is more than one background image after exposure   
    if len(frIndicesBgPost) > 1:
        bgImgPost = np.mean(
        frames[frIndicesBgPost[0]:
        frIndicesBgPost[1]], axis = 0)
            
    # If there is only one background image after exposure
    if len(frIndicesBgPost) == 1:
        bgImgPost = frames[frIndicesBgPost]
    
    # If there is no background image after exposure
    if len(frIndicesBgPost) == 0:
        bgImgPost = 0

    print("%i frames pre exposure, from index %i to %i" %(numOfBgFramesPre,
            frIndicesBgPre[0],frIndicesBgPre[-1]))
    print("%i exposed frames, from index %i to %i" %(numOfExpFrames,
                                frIndicesBeamOn[0], frIndicesBeamOn[-1]))                                    
    print("%i frames post exposure" %(numOfBgFramesPost))
    
    # Checking for violation of ADC limit
    for index in range(len(frIndicesBeamOn)-1):
        if countPxlOverViolLimit[
                frIndicesBeamOn[index]] > tolAmountOfADCLimitViol:
            print("### ADC limit reached ###") 
    
    ## Perform I. background, II. bad pixel and III. sensitivity correction
    # I. Background correction
    # Calculate temporal mean of background
    bgImgPre = np.mean(frames[frIndicesBgPre], axis = 0)
    bgImgPreSTD = np.std(frames[frIndicesBgPre], axis = 0)
    
    # Perform background subtraction  
    bgCorrImgContainer = frames - bgImgPre
    print("Background correction done...")
    
    # Create background corrected unexposed frames
    tempMeanOfBgCorrBg = np.mean(bgCorrImgContainer[frIndicesBgPre], axis = 0)
    tempNoisePerPxl = np.std(bgCorrImgContainer[frIndicesBgPre], axis = 0)
    bgCorrBgSummedUp = np.sum(bgCorrImgContainer[frIndicesBgPre], axis = 0)    
        
    # Find pixel above threshold = exposed frame
    bgCorrThres = bgCorrImgContainer > (
    tempMeanOfBgCorrBg + integrationThresFactor * tempNoisePerPxl)
    
    # Choose only pixel above threshold for summation
    bgCorrImgContainer = bgCorrImgContainer * bgCorrThres
    
    # Sum up exposed frames over threshold
    bgCorrImg = np.sum(bgCorrImgContainer[frIndicesBeamOn], axis = 0)
    
    # II. Bad pixel correction, neighborhood-averaging from Christian Lampe
    # Create bad pixel matrix
    # Calculate threshold for bad pixel definition based on to background image
    permBadPxlMatrixHigh = bgImgPre > (np.median(
    bgImgPre) + badPxlFactor * np.std(bgImgPre))
    permBadPxlMatrixLow = bgImgPre < (np.median(
    bgImgPre) - badPxlFactor * np.std(bgImgPre))
    
    if np.sum(bgImgPost) == 0:
             tempBadPxlMatrix = 0
    else:
        tempBadPxlMatrix = bgImgPost > bgImgPre + np.std(bgImgPre)
    
    badPxlMatrix = permBadPxlMatrixHigh + permBadPxlMatrixLow + tempBadPxlMatrix
    numBadPxl = np.sum(badPxlMatrix) / ((int(numOfPxlDim[0])*int(numOfPxlDim[1])))
    print("%0.1f %% pixel are out of tolerance and will be corrected."
          %(numBadPxl*100))
    
    # Apply neighborhood filtering
    rowy, coly = bgCorrImg.shape
     
    # Apply zero-padding
    corrImg = np.zeros([rowy+2, coly+2])
    corrImg[1:rowy+1, 1:coly+1] = bgCorrImg
    
    newMatrix = np.zeros([rowy+2, coly+2])
    newMatrix[1:rowy+1, 1:coly+1] = bgCorrImg
    
    badPxlMap = np.zeros([rowy+2, coly+2])
    badPxlMap[1:rowy+1, 1:coly+1] = badPxlMatrix
    
    # Define pixel value "0" as bad pixel
    badPxlMap = 1 - badPxlMap
    
    for row in range(1,rowy+1,1):
        for col in range(1,coly+1,1):
            if badPxlMap[row, col] == 0:
                nb1 = newMatrix[row-1, col-1] * badPxlMap[row-1,col-1]
                nb2 = newMatrix[row-1, col] * badPxlMap[row-1,col]
                nb3 = newMatrix[row-1, col+1] * badPxlMap[row-1,col+1]
                nb4 = newMatrix[row, col-1] * badPxlMap[row,col-1]
                nb5 = newMatrix[row, col+1] * badPxlMap[row,col+1]
                nb6 = newMatrix[row+1, col-1] * badPxlMap[row+1,col-1]
                nb7 = newMatrix[row+1, col] * badPxlMap[row+1,col]
                nb8 = newMatrix[row+1, col+1] * badPxlMap[row+1,col+1]
                nbVector = [nb1, nb2, nb3, nb4, nb5, nb6, nb7, nb8]
                validPxl = np.sum(badPxlMap[row-1:row+1, col-1:col+1])
                meanPxlValue = np.median(nbVector)
                if validPxl == 0:
                    corrImg[row, col] = corrImg[row, col] 
                elif validPxl == 1:
                    corrImg[row, col] = meanPxlValue
                elif validPxl == 2:
                    corrImg[row, col] = meanPxlValue
                elif validPxl == 3:
                    corrImg[row, col] = meanPxlValue
                elif validPxl == 4:
                    corrImg[row, col] = meanPxlValue
                elif validPxl == 5:
                    corrImg[row, col] = meanPxlValue
                elif validPxl == 6:
                    corrImg[row, col] = meanPxlValue
                elif validPxl == 7:
                    corrImg[row, col] = meanPxlValue
                elif validPxl == 8:
                    corrImg[row, col] = meanPxlValue
     
    badPxlCorrImg = corrImg[1:1025,1:1025]
    badPxlMatrix = badPxlMatrix[1:1025,1:1025]
    print("Bad pixel correction done...")
    
    ## III. Pixel sensitvity correction
    # Perform sensitivity correction
    imgResult = pxlGainMatrix * badPxlCorrImg
    print("Sensitivity correction done...") 
    
    # Apply median filter
    procImage = signal.medfilt2d(imgResult,medFiltKernelSize)
    
    if medFiltKernelSize > 1:
        print("Median filter of %1.1f mm² kernel size applied"
              %(medFiltKernelSize*medFiltKernelSize*pxlShiftInMm*pxlShiftInMm))
    else:
        print("No median filter applied...")
    
    # Rotate image according to machine beam coordinate system, beams eye view
    procImage = np.rot90(procImage,3)
    procImage = np.flipud(procImage)
    
    ## ## ## NOISE CALCULATION ## ## ##
    ###################################
    
    # Calculate temporal noise based on background corrected background images ... TO BE CHECKED
    noiseStatistics = [np.min(tempNoisePerPxl),
                           np.mean(tempNoisePerPxl),
                           np.std(tempNoisePerPxl),
                           np.median(tempNoisePerPxl),
                           np.max(tempNoisePerPxl)]
    
    noise = noiseStatistics[1] + (5 * noiseStatistics[2])
    
    print("Pixel noise: %i counts" %(noise))

    # Calculate effective noise based on the number of exposed frames
    effNoise = (noise) * np.sqrt(numOfExpFrames)
    print("Effective noise: %i counts" %(effNoise))

    # Sum first few background corrected exposed frames
    # Estimate average exposure signal intensity
    sigIntArray = []
    for elements in range(len(frIndicesBeamOn)):    
        sigSurArray = bgCorrImgContainer[frIndicesBeamOn[elements]]
        # Median filter the array to get rid of stinky pixel
        sigSurArrayFilt = signal.medfilt2d(sigSurArray,3)
        centOfMass = scipy.ndimage.center_of_mass(sigSurArrayFilt)
        if np.isnan(centOfMass[0]) or np.isnan(centOfMass[1]) == True:
            break
        sigIntensity = sigSurArrayFilt[
        np.int64(centOfMass[0]),np.int64(centOfMass[1])]
        sigIntArray.append(sigIntensity)
        
    # Calcultate average background corrected signal    
    sigIntensity = np.median(sigIntArray)
    
    # Calculate average background signal
    avgBackground = np.median(bgImgPre)
    
    # If no center of mass can be calculated
    if np.isnan(sigIntensity) == True:
        sigIntensity = np.mean(bgCorrBgSummedUp)
    
    # Calculate signal to noise ration (SNR) 
    SNR = sigIntensity / effNoise
    
    if np.isnan(avgBackground) == 0:
        avgBackground == np.mean(bgCorrBgSummedUp)
    
    
    # Calculate % full scale range
    FSR = (sigIntensity + avgBackground) / bitResolution
       
    print("Signal: %i counts, average background: %i counts, SBR: %i"
          %(sigIntensity+avgBackground, avgBackground, (sigIntensity+avgBackground) / avgBackground))
    
    print("Signal to noise ratio (SNR): %0.1f" %(SNR))
    
    print("%0.1f %% of FSR" %(FSR*100))

    # Write image aquisition infos to textfile
    with open(date + "_process" + ".log", "a") as f:
           f.write("############ %s #############\n" %dateTime)
           f.write("%s\n" %(file)) 
           f.write("Total number of frames: %i\n" %numOfFrames)
           f.write("Total number of valid frames: %i\n" %lastFrameIndex)
           f.write("%i frames pre exposure, from index %i to %i\n"
            %(numOfBgFramesPre, frIndicesBgPre[0],frIndicesBgPre[-1]))
           f.write("%i exposed frames from index %i to %i\n" 
            %(numOfExpFrames, frIndicesBeamOn[0], frIndicesBeamOn[-1]))
           f.write("%i frames post exposure\n" 
            %(numOfBgFramesPost))
           f.write("%0.1f %% of pixel out of tolerance to be corrected\n"
            %(numBadPxl*100))
           f.write("Median filter of %1.1f mm² kernel size applied\n"
            %(medFiltKernelSize*medFiltKernelSize*pxlShiftInMm*pxlShiftInMm))
           f.write("Pixel noise: %0.1f counts\n"
            %(noiseStatistics[2]))
           f.write("Effective noise: %0.1f counts\n" %(effNoise))
           f.write("Signal: %i counts, background: %i counts, Signal to background ratio (SBR): %i\n"
            %(sigIntensity+avgBackground, avgBackground, (sigIntensity+avgBackground) / avgBackground))
           f.write("Signal to noise ratio (SNR): %0.2f\n"
            %(SNR))
           f.write("%0.1f %% FSR\n\n" %(FSR*100))
           f.close

    # Return processed image
    imgQual = [SNR, FSR]
    imgs2Corr = [bgImgPre, bgImgPost, pxlGainMatrix, badPxlMatrix]
    
    return procImage, imgQual, pxlShiftInMm, imgs2Corr, bg

def writeTIFF(procImage, SNR, FSR, file):
    # Create 16 bit unsigned version of the image
    procImage16bit = procImage
    maxValue = np.max(procImage)
    procImage16bit = procImage/maxValue * 65535
    convFactor = maxValue / 65535
    procImage16bit = np.uint16(procImage16bit)
    imwrite(file + ".tiff", procImage16bit,
            resolution = (5 * 25.4, 5 * 25.4),
            metadata={
                "ConvertTo32BitFactor": convFactor, 
                "%FSR": int(FSR*100),
                "SNR": SNR
                })
    return
    

