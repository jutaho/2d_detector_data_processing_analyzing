#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Description -     analyzes images regarding flatness
# Last change -     24.04.2024
# Author -          J. Horn

# Import modules
import datetime
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal import savgol_filter

def analyze(img, pxlShiftInMm, fileName, detector):
    # Define variables
    date = datetime.datetime.now()
    format = "%Y-%m-%d %H:%M:%S"
    date = date.strftime(format)
    interpFactor = 0.002
    fieldsizeThreshold = 0.5
    penumbraThresholdLow = 0.2
    penumbraThresholdHigh = 0.8
    evalAreaFactor = 4
    if detector == "flatpanel":
            coord = [-102.4, 102.4, -102.4, 102.4]
            lengthArea2NormInPxl = 100
    elif detector =="reference":
            coord = [-125, 125, -125, 125]
            lengthArea2NormInPxl = 100
    else:
            coord = [-75, 75, -75, 75]
            lengthArea2NormInPxl = 5
    
    # Define image properties 
    # Caluclate image origin
    sizeImgVert, sizeImgHoriz = img.shape
    originVert = np.round(sizeImgVert/2)
    originHoriz = np.round(sizeImgHoriz/2) 
    
    # Determine the middle of the exposed area  
    # Projection in both directions
    # Vectors -> 0 = rows, 1 = cols
    medProjImgVert = np.sum(img, axis = 0)
    medProjImgHoriz = np.sum(img, axis = 1)
    
    # Differentiate both projections 
    numDiffProjImgVert = np.gradient(medProjImgVert)
    numDiffProjImgHoriz = np.gradient(medProjImgHoriz)
    
    # Filter both projections
    filtNumDiffProjImgVert = savgol_filter(numDiffProjImgVert,5,3)
    filtNumDiffProjImgHoriz = savgol_filter(numDiffProjImgHoriz,5,3) 
    
    # Find the positions of both inflection points = Full Width Half Maximum
    # Interpolate both filtered projections
    xVert = np.arange(len(filtNumDiffProjImgVert))
    xHoriz = np.arange(len(filtNumDiffProjImgHoriz))
    
    # Interpolate 
    xInterVer = np.arange(0, xVert[-1], interpFactor)
    yInterVer = np.interp(xInterVer, xVert, filtNumDiffProjImgVert)
    
    xInterHor = np.arange(0, xHoriz[-1], interpFactor)
    yInterHor = np.interp(xInterHor, xHoriz, filtNumDiffProjImgHoriz)
    
    # Look for maxima and minima = inflections points
    # Fist and last 10 pxl to eliminate potential artifacts 
    posInflPntVert1 = np.concatenate(np.where(
        yInterVer == np.max(yInterVer)))
    posInflPntVert2 = np.concatenate(np.where(
        yInterVer == np.min(yInterVer)))
    
    pxlPosCentIrradFieldVert = np.round(
    (posInflPntVert1 + posInflPntVert2) / 2 * interpFactor)
    
    posInflPntHoriz1 = np.concatenate(np.where(
        yInterHor == np.max(yInterHor)))
    posInflPntHoriz2 = np.concatenate(np.where(
        yInterHor == np.min(yInterHor)))
    
    pxlPosCentIrradFieldHoriz = np.round(
    (posInflPntHoriz1 + posInflPntHoriz2) / 2 * interpFactor)
    
    # Calculate normalization value = median
    normValue = np.median(
        img[int(pxlPosCentIrradFieldVert-lengthArea2NormInPxl):
                int(pxlPosCentIrradFieldVert+lengthArea2NormInPxl),
                int(pxlPosCentIrradFieldHoriz-lengthArea2NormInPxl):
                int(pxlPosCentIrradFieldHoriz+lengthArea2NormInPxl)])
        
    # Normalize to median of "dose"-distribution    
    imgRel = img / normValue

    # Determine penumbra and fieldsize according to central profiles
    # Determine central profiles
    profileVert = imgRel[:,int(pxlPosCentIrradFieldHoriz)]
    profileHoriz = imgRel[int(pxlPosCentIrradFieldVert),:]
    
    # Determine x-vectors for both profiles
    posVectorVert = (np.arange(len(profileVert)) - pxlPosCentIrradFieldVert) 
    posVectorHoriz = (np.arange(len(profileHoriz)) - pxlPosCentIrradFieldHoriz)
    
    posVectorVertInMm = posVectorVert * pxlShiftInMm
    posVectorHorizInMm = posVectorHoriz * pxlShiftInMm
    
    # Interpolate profiles
    posVectorVertInter = np.arange(posVectorVert[0],
                                   posVectorVert[-1], interpFactor)
    profileVertInter = np.interp(posVectorVertInter,
                                 posVectorVert, profileVert)
    
    posVectorHorizInter = np.arange(posVectorHoriz[0],
                                    posVectorHoriz[-1], interpFactor)
    profileHorizInter = np.interp(posVectorHorizInter,
                                  posVectorHoriz, profileHoriz)
    
    # Calculate fieldsizes
    thresVert = profileVertInter >= fieldsizeThreshold
    vertPositions = np.concatenate(np.where(thresVert == 1))
    fieldsizeVert = (vertPositions[-1] - vertPositions[0]) * interpFactor
    fsVertInMm = fieldsizeVert * pxlShiftInMm
    
    thresHoriz = profileHorizInter >= fieldsizeThreshold
    horizPositions = np.concatenate((np.where(thresHoriz == 1)))
    
    fieldsizeHoriz = (horizPositions[-1] - horizPositions[0]) * interpFactor
    fsHorizInMm = fieldsizeHoriz * pxlShiftInMm
    
    # Calculate horizontal penumbra
    thres20 = profileHorizInter >= penumbraThresholdLow
    thres80 = profileHorizInter >= penumbraThresholdHigh
    thres20 = np.concatenate(np.where(thres20 == 1))
    thres80 = np.concatenate(np.where(thres80 == 1))
    
    penumbraL = (thres80[0] - thres20[0]) * interpFactor 
    penumbraR = (thres20[-1] - thres80[-1]) * interpFactor
    
    # Calculate vertical penumbra
    thres20 = profileVertInter >= penumbraThresholdLow
    thres80 = profileVertInter >= penumbraThresholdHigh
    thres20 = np.concatenate(np.where(thres20 == 1))
    thres80 = np.concatenate(np.where(thres80 == 1))
    
    penumbraU = int(thres80[0] - thres20[0]) * interpFactor
    penumbraD = int(thres20[-1] - thres80[-1]) * interpFactor

    # Check for minimal penumbra and fieldsize
    minPenumbra = np.min([penumbraL,penumbraR,penumbraU,penumbraD])
    minFieldsize = np.min([fieldsizeVert,fieldsizeHoriz])
    
    # Calculate CAX in "mm", SIGNS need to be checked
    caxHoriz = (pxlPosCentIrradFieldHoriz - (originHoriz)) * pxlShiftInMm 
    caxVert = (pxlPosCentIrradFieldVert - (originVert)) * pxlShiftInMm
    
    # Define area of evalutation
    evalRange = int(minFieldsize - (evalAreaFactor * (minPenumbra)))
    
    horizStart = int(pxlPosCentIrradFieldHoriz - (evalRange / 2))
    horizEnd = int(pxlPosCentIrradFieldHoriz + (evalRange / 2))
    HORIZ = np.arange(horizStart,horizEnd)
  
    
    vertStart = int(pxlPosCentIrradFieldVert - (evalRange / 2)) 
    vertEnd = int(pxlPosCentIrradFieldVert + (evalRange / 2)) 
    VERT = np.arange(vertStart,vertEnd)
    
    # Calculate relevant statistics within area of evaluation = ROI
    Dmax = np.max(imgRel[VERT[0]:VERT[-1],HORIZ[0]:HORIZ[-1]])
    Dmin = np.min(imgRel[VERT[0]:VERT[-1],HORIZ[0]:HORIZ[-1]])
    Drange = Dmax - Dmin
    Dmean = np.mean(imgRel[VERT[0]:VERT[-1],HORIZ[0]:HORIZ[-1]])
    Dsd = np.std(imgRel[VERT[0]:VERT[-1],HORIZ[0]:HORIZ[-1]])
    
    # Calculate "inter quartile ranges" from 0.01 to 0.99 of all pixel values
    q1 = 0.001 
    q2 = 1 - q1
    twoDimArray = imgRel[VERT[0]:VERT[-1],HORIZ[0]:HORIZ[-1]]
    sortedArray = np.sort(twoDimArray.flatten())
    
    qa = sortedArray[int(np.round(q1*len(sortedArray)))]
    qb = sortedArray[int(np.round(q2*len(sortedArray)))]
    D_qb_minus_qa = (qb - qa)
    D_IQR_percent = (D_qb_minus_qa/2) * 100
    
    # Calculate 2D-Flatness
    FlatnessTG224 = (Dmax - Dmin) / (Dmax + Dmin)
    FlatnessPercent = FlatnessTG224 * 100

    print("\n2D-Flatness: %0.1f %%..." %(FlatnessPercent))    

    # Plot relevant data
    # Show image of relative dose distribution
    Rows, Cols = np.shape(img)
    img2dPlot = np.ones([Rows, Cols])
    img2dPlot = img2dPlot * imgRel
    
    # Contouring ROI
    img2dPlot[VERT[0]:VERT[-1],HORIZ[0]:HORIZ[1]] = 0
    img2dPlot[VERT[0]:VERT[-1],HORIZ[-2]:HORIZ[-1]] = 0
    img2dPlot[VERT[0]:VERT[1],HORIZ[0]:HORIZ[-1]] = 0
    img2dPlot[VERT[-2]:VERT[-1],HORIZ[0]:HORIZ[-1]] = 0
    img2dPlot[int(pxlPosCentIrradFieldVert),:] = 0
    img2dPlot[:,int(pxlPosCentIrradFieldHoriz)] = 0
    
    # Plot 2d-image 
    fig, axes = plt.subplots(figsize = (10,10), dpi = 150) 
    
    plt.rcParams['axes.titlesize'] = 12
    axes.tick_params(axis='x', labelsize = 12)
    axes.tick_params(axis='y', labelsize = 12)
    
    imgHandle = axes.imshow(
    img2dPlot, cmap = plt.cm.bone, vmin = 0.8, vmax = 1.2,
    origin = "lower", extent = coord)
    axes.set( 
    title ="2D-Flatness within ROI = $\pm$ %0.1f%%" %(
    FlatnessPercent))
    
    axes.set_xlabel("horizontal Position [mm]", fontsize = 12) 
    axes.set_ylabel("vertical Position [mm]", fontsize = 12)
    cbar = fig.colorbar(imgHandle, shrink = 0.5)
    cbar.set_label("normalized Signal", fontsize = 12)
    cbar.ax.tick_params(labelsize = 12)
    axes.set_xlim(-75, 75)
    axes.set_ylim(-75, 75)
    
    
    # Create and plot histogram
    # Create Object for subplot histogram and central profiles
    fig1, axes1 = plt.subplots(2,1, figsize = (10,10), dpi = 150)
    
    plt.rcParams['axes.titlesize'] = 12
    axes.tick_params(axis='x', labelsize = 12)
    axes.tick_params(axis='y', labelsize = 12)
    
    # numberOfBins = max - min / bin width
    nBins = int((max(sortedArray) - min(sortedArray)) / 0.001) 
    counts, bins = np.histogram(sortedArray, bins = nBins) 
    axes1[0].hist(sortedArray, bins, weights = np.ones_like(
    sortedArray)/len(sortedArray)*100,
    color = "white", ec = "black")
    axes1[0].vlines(x = Dmean,
    ymin = 0, ymax = 20, color = "black", linestyle = "-")
    axes1[0].vlines(x = [Dmean + Dsd, Dmean - Dsd], ymin = 0, ymax = 20,
    color = "green", linestyle = "--")
    axes1[0].vlines(x = [Dmax, Dmin],
    ymin = 0, ymax = 20, color = "red", linestyle = ":")
    axes1[0].vlines(x = [0.95, 1.05], ymin = 0, ymax = 20, color ="red")
    axes1[0].set(title ="Histogram of normalized Signal within ROI")
    axes1[0].set_xlabel("normalized Signal", fontsize = 12)     
    axes1[0].set_ylabel("Frequency [%]", fontsize = 12)
    axes1[0].set(xlim = (0.94, 1.06))
    axes1[0].set_xticks(
    [0.94,0.95,0.96,0.97,0.98,0.99,1.00,1.01,1.02,1.03,1.04,1.05, 1.06])  
    axes1[0].set(ylim = (0, 10))
    axes1[0].legend(
    ["$\mu$ = %0.2f" %Dmean,"$\sigma$ = $\pm$ %0.2f" %Dsd,
     "Dmax - Dmin = %0.2f" %Drange,
     "Tolerance-Levels"], loc="upper right", framealpha = 1)
    
    # Plot central line profiles
    axes1[1].plot(
    posVectorVertInMm,profileVert,"g.",posVectorHorizInMm,profileHoriz,"b.")
    axes1[1].set(title ="Central Line Profile")
    axes1[1].set_xlabel("Position [mm]", fontsize = 12)     
    axes1[1].set_ylabel("normalized Signal", fontsize = 12) 
    axes1[1].set(ylim = (0.5, 1.2))
    axes1[1].set(xlim = (-75, 75))
    axes1[1].legend(["vertical Fieldsize = %0.1f mm"
                     %(fsVertInMm),
                  "horizontal Fieldsize = %0.1f mm"
                  %(fsHorizInMm)],
                 loc="center", framealpha = 1)
    axes1[1].axhline(0.95, color = "red")
    axes1[1].axhline(1.05, color = "red")
    axes1[1].grid(True)  
    
    
    # Plot multiple profiles
    # Create object for subplot of multiple profiles
    #fig2, axes2 = plt.subplots(figsize = (7,3.5), dpi = 300)
    fig2, axes2 = plt.subplots(2,1, figsize = (10,10), dpi = 150)

    posHoriz = (HORIZ - pxlPosCentIrradFieldHoriz) * pxlShiftInMm
    posVert = (VERT - pxlPosCentIrradFieldVert) * pxlShiftInMm

    plt.rcParams['axes.titlesize'] = 12
    plt.rc('xtick',labelsize=12)
    plt.rc('ytick',labelsize=12)

    # Plot verticals
    axes2[0].plot(posHoriz, imgRel[VERT, HORIZ[0]:HORIZ[-1]])
    axes2[0].set(title ="Line profiles")
    axes2[0].set_xlabel("vertical Position [mm]", fontsize = 12)     
    axes2[0].set_ylabel("normalized Signal", fontsize = 12)  
    axes2[0].set(ylim = (0.8, 1.2))
    axes2[0].set(xlim = (-75, 75)) 
    axes2[0].axhline(0.95, color = "red")
    axes2[0].axhline(1.05, color = "red")
    axes2[0].grid(True)
    
    # Transpose matrix to plot multiple horizontal profiles
    
    m = np.matrix(imgRel)
    imgRelTrans = m.transpose()
    axes2[1].plot(posVert, imgRelTrans[HORIZ, VERT[0]:VERT[-1]])
    axes2[1].set_xlabel("horizontal Position [mm]", fontsize = 12)     
    axes2[1].set_ylabel("normalized Signal", fontsize = 12)
    axes2[1].set(ylim = (0.8, 1.2))
    axes2[1].set(xlim = (-75, 75))
    axes2[1].axhline(0.95, color = "red")
    axes2[1].axhline(1.05, color = "red")
    axes2[1].grid(True)
    

    # Take figrues and write PDF report
    plt.rcParams["figure.figsize"] = [11.69, 8.268]
    plt.rcParams["figure.autolayout"] = False
    horizLine = "__________________________________________________"
    fileName = fileName
    p = PdfPages(fileName)
    fig_nums = plt.get_fignums()  
    figs = [plt.figure(n) for n in fig_nums]
    totNbPages = len(figs)
    count = 0
    
    for fig in figs:
        count = count + 1
        fig.text(0.05,0.99,horizLine)
        fig.text(0.05,0.97, fileName)
        fig.text(0.05, 0.02, "%s" %date)
        fig.text(0.8,0.02, "Page %i of %i" %(count,totNbPages))
        fig.savefig(p, format="pdf", dpi = 150)
        plt.close()
    p.close()
