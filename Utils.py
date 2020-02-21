import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import statistics as stat
import math
import os
import glob
import networkx as nx
from skimage.filters import (threshold_otsu, threshold_sauvola)
from skimage.transform import hough_line
from skimage import measure
from sklearn.preprocessing import binarize
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.neighbors import kneighbors_graph
from collections import defaultdict  
from scipy.signal import find_peaks
from PIL import Image
import pytesseract
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
import argparse
import heapq
import PreProcessing as pp
import LayoutAnalysis as la

def showImage(text_name, file_name):
    '''
    It shows the image creating a new OpenCV window, it also write the image on a file.
    
    Parameters
    ----------
    text_name : string about the file named used to show and write
    file_name: array of image pixels to show.
    '''
    cv.namedWindow(text_name, cv.WINDOW_NORMAL)
    cv.imshow(text_name, file_name)
    cv.imwrite(text_name+'.tif', file_name)
    cv.waitKey(0)
    cv.destroyWindow(text_name)
    
def removeFiguresOrSpots(binarized_img, mode):
    '''
    Remove figures or spots from images, useful for making more accurate the deskew with Hough transform.
    
    Parameters
    ----------
    binarized_img : array of binarized image pixels.
    original_img: array of original image pixels.
    mode : string used for selection of mode. 

    Returns
    -------
    new_img : array of original image pixels without figures or spots.
    '''
    if mode != 'figures':
        kernel = np.ones((3,2), np.uint8) 
        binarized_img = cv.dilate(~binarized_img, kernel, iterations=1)
        binarized_img = cv.erode(binarized_img, kernel, iterations=1)
        binarized_img = ~binarized_img
    new_img = binarized_img.copy()
    if new_img.ndim >2:
        new_img = cv.cvtColor(new_img, cv.COLOR_BGR2GRAY)
    contours,_ = cv.findContours(~binarized_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    
    for contour in contours:
        [x,y,w,h] = cv.boundingRect(contour) #posso usare cv.contourArea(contour)
        if mode == 'figures':
            if w>300 or h>300 :#remove if the box it's too big (figure) or if it's too small (spot)
                for i in range(y,y+h):
                    for j in range(x, x+w):
                        new_img[i][j] = 255
        elif mode == 'spots':
            if ((0<=w<=5) and (0<=h<=5)):  #DACONTROLLARE (RIMUOVE PUNTI)
                for i in range(y,y+h):
                    for j in range(x, x+w):
                        new_img[i][j] = 255
        elif mode == 'linesVert':
            if h>50 and w<25:
                for i in range(y,y+h):
                    for j in range(x, x+w):
                        new_img[i][j] = 255
        elif mode == 'linesHoriz':
            if w>50 and h<25:
                for i in range(y,y+h):
                    for j in range(x, x+w):
                        new_img[i][j] = 255
        elif mode == 'linesBoth':
            if (w>50 and h<25) or (h>50 and w<25):
                for i in range(y,y+h):
                    for j in range(x, x+w):
                        new_img[i][j] = 255
                        
    #showImage('gn',new_img)
    return new_img

def printContours(binarized_img,output_img, thickness):
    '''
    Draw a green rectangle (if the image is not in grayscale) around to characters/words.
    
    Parameters
    ----------
    binarized_img : array of binarized image pixels.
    output_img: array of output image pixels. This will be modified at the end of the method.
    '''
    box_coordinates = []
    #contours,_  = cv.findContours(np.uint8(np.logical_not(binarized_img)),cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE) 
    contours,_  = cv.findContours(~binarized_img,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE) 
    for contour in contours:
        [x,y,w,h] = cv.boundingRect(contour)
        if w >= 4 and h >= 4:
            cv.rectangle(output_img, (x,y), (x+w,y+h), (0, 255, 0), thickness)
        if w >= 100 and h >= 100:
            box_coordinates.append([x,y,w,h])
    box_coordinates = sorted(box_coordinates, key = lambda z:z[1])
    box_coordinates = sorted(box_coordinates, key = lambda z:z[0])
    return box_coordinates
            
            
            
def findMidDistanceContour(binarized_img,vert: bool = False):
    contours,_  = cv.findContours(~binarized_img,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE) 
    sumDir = 0
    size = 0
    for contour in contours:
        [x,y,w,h] = cv.boundingRect(contour)
        if vert:
            sumDir += h
        else:
            sumDir += w
        size += 1
    midV = sumDir/size
    return midV
    



#trova la distanza tra centroidi
def findDistance(binarized_img, vert: bool = False):
    points = la.findCentroids(binarized_img,binarized_img.copy())
    
    G,edges = la.minimumSpanningTreeEdges(points,5)
    edges_dist = G.data
    edges_hor,edges_vert = edgesInformation(edges,points,edges_dist)
    distances = []
    edges = edges_hor
    if vert:
        edges = edges_vert
    for edge in edges:
        c1,c2 = edge[2]
        x1,y1 = points[c1]
        x2,y2 = points[c2]
        dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        distances.append(dist)
       
    #DA VERIFICARE MEGLIO
    for dist in distances:
        if dist > 150:
            distances.remove(dist)
    
    
    return distances


def histogram(binarized_img,distance,vert: bool = False):
    w=findMidDistanceContour(binarized_img,vert)
    row_number = [i for i in range(len(distance))]
    distances=[]
    for j in range(len(distance)):
        distances.append(distance[j]-w)
    plt.bar(row_number,distances)
    plt.show()        
              
    
def rotate(img_orig,img_bin):
    img_copy = img_orig.copy()
    printContours(img_bin,img_copy, 1)       

    valueH,_= pp.valueRLSA(img_bin)
    valueV,_ = pp.valueRLSA(img_bin,True)
    
    img_rlsa_H = pp.rlsa(img_bin.copy(), True, False, valueH)
    img_rlsa_full = pp.rlsa(img_rlsa_H.copy(),False,True,valueV)
    
    img_no_figures = removeFiguresOrSpots(img_rlsa_full, 'figures')
    #showImage('puppu',img_no_figures)
    img_rotated,_ = pp.houghTransformDeskew(img_no_figures, img_orig, False)
    return img_rotated

def showProjection(img_bin, counts, row_number):
    
    f, (ax1,ax2)= plt.subplots(1, 2, sharey=True, figsize=(70, 20))#(70,40)
    ax1.imshow(img_bin,'gray')
    ax1.tick_params(axis='both', which='major', labelsize=30)
    ax2.plot(counts, row_number,label='fit')
    ax2.tick_params(axis='both', which='major', labelsize=30)
    plt.xlabel('Number of Black Pixels',fontsize=40)
    plt.ylabel('Row Number',fontsize=40)
    plt.subplots_adjust( wspace = 0)
    plt.show()
    
    
def rectContains(rect, point) :
    '''
    Calculates the belonging of a point to a given rectangle.
    
    Parameters
    ----------
    rect: tuple that contains origin and size of the image rectangle
    point : coordinate (x, y) of a point
    
    Returns
    -------
    contains: boolean value
    '''
    if point[0] < rect[0] :
        return False
    elif point[1] < rect[1] :
        return False
    elif point[0] > rect[2] :
        return False
    elif point[1] > rect[3] :
        return False
    return True

def euclidean_distance(point1, point2):
	distance = (point2[0] - point1[0])**2 + (point2[1] - point1[1])**2
	return math.sqrt(distance)

def angleBetween(p1, p2):
    dx = p2[1]-p1[1]
    dy = p2[0]-p1[0]
    arctan = math.atan2(dx,dy)
    return math.degrees(arctan)

def getAngles(edges,points):
    angles = []
    for edge in edges:
        #i, j = edge
        i,j = edge
        angles.append(angleBetween(points[i],points[j]))
    return angles

def plotEdges(img, edges, points):
    points = np.int32(points)
    output_img = img.copy()
      
    for edge in edges:
        
        i,j = edge
        cv.line(output_img, (points[i, 0], points[i, 1]), (points[j, 0], points[j, 1]), (0,0,0), 1, cv.LINE_AA)
         
    return output_img

def edgesInformation(edges, points, distances):
    angles = getAngles(edges,points)
    points = np.int32(points)

    horizontal_edges =[]
    vertical_edges = []
    
    for i in range(len(angles)):
    
        #if -15< angles[i] < 15 or 165 < angles[i] or angles[i] < -165:
        if -20< angles[i] < 20 or 160 < angles[i] or angles[i] < -160:
            horizontal_edges.append((angles[i],distances[i],[edges[i][0],edges[i][1]]))
        elif 70 < angles[i] < 110 or (-70 > angles[i] and angles[i] > -110)  :
            vertical_edges.append((angles[i],distances[i],[edges[i][0],edges[i][1]]))
    
    return horizontal_edges,vertical_edges

def polyArea(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def counterClockwise(A,B,C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

# Return true if line segments AB and CD intersect
def intersect(A,B,C,D):
    return counterClockwise(A,C,D) != counterClockwise(B,C,D) and counterClockwise(A,B,C) != counterClockwise(A,B,D)  


def findPeaks(k_kneighbors_distances, distance: int=0, plot: bool=False):
    d = defaultdict(int)
    
    for k in k_kneighbors_distances:
        k = round(k)
        d[k] += 1
        
    result = sorted(d.items(), key = lambda x:x[1], reverse=True)
    
    values = []
    occurrences = []
    for i in range(len(result)):
        occurrences.append(result[i][1])
        values.append(result[i][0])

    x = np.array(values)
    y = np.array(occurrences)
    
    # orina i dati dal valore più grande al valore piu piccolo(sia rispetto a x sia rispetto a y)
    # x = [8, 3, 5] diventa x = [3, 5, 8]
    sortId = np.argsort(x)
    x = x[sortId]
    y = y[sortId]

    #Trovo i punti di ottimo locale
    if(distance!=0):
        peaks, _ = find_peaks(y, distance= distance)
    else:
        peaks, _ = find_peaks(y)


    peaksOccurrenceList=[]
    peaksOccurrences=[]
    for i in range(len(peaks)):
        peaksOccurrences.append(y[peaks[i]])
        peaksOccurrenceList.append((x[peaks[i]],y[peaks[i]]))
    peaksOccurrenceList = sorted(peaksOccurrenceList, key = lambda x:x[1], reverse=True)
    
    #Trova i DUE picchi piu alti 
    bestPeaks=heapq.nlargest(2, peaksOccurrences)


    my_Peak_Value=[]
    for k in range(len(peaksOccurrenceList)):
        if bestPeaks[0]==peaksOccurrenceList[k][1] or bestPeaks[1]==peaksOccurrenceList[k][1]:
            my_Peak_Value.append(int(peaksOccurrenceList[k][0]))  
    
    
    if plot:
        plt.plot(x, y)
        plt.plot(my_Peak_Value, bestPeaks, "x")
        plt.xlim(0, 80)
        plt.show()
        print('Peak values for this image are',my_Peak_Value)
    return my_Peak_Value