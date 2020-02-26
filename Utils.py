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
#Use this string on Windows OS
#pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
import argparse
import heapq
import PreProcessing as pp
import LayoutAnalysis as la

def showImage(text_name, image, img_name: str='', out_path: str='', write: bool=False):
    '''
    It shows the image creating a new OpenCV window, it also write the image on a file 
    whose extension can be specified in the constant IMAGE_EXTESION.
    
    Parameters
    ----------
    text_name : string about the file named used to show and write
    image: array of image pixels to show.
    img_name: string of name file that will be created (optional if you want only to show image)
    out_path: string of the path where file will be created (optional if you want only to show image)
    write: boolean value to enable the write function
    
    '''
    IMAGE_EXTESION = '.png'
    cv.namedWindow(text_name, cv.WINDOW_NORMAL)
    cv.imshow(text_name, image)
    if write:
        cv.imwrite(out_path+img_name+ ' ' +text_name+ IMAGE_EXTESION, image)
    cv.waitKey(0)
    cv.destroyWindow(text_name)
    
def removeFiguresOrSpots(binarized_img, mode):
    '''
    Remove figures,spots and lines from an image, useful for making more accurate the deskew with Hough transform and
    for help to compute several function.
    If the function is remove spots or lines it makes a dilatation and erosion to increase accuracy.
    
    Parameters
    ----------
    binarized_img : array of binarized image pixels.
    mode : string used for selection of mode. 

    Returns
    -------
    new_img : array of original image pixels with elements removed.
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
        [x,y,w,h] = cv.boundingRect(contour)
        
        if mode == 'figures':
            if (w>250 or h>250):#remove if the box it's too big (figure) 
                for i in range(y,y+h):
                    for j in range(x, x+w):
                        new_img[i][j] = 255
        elif mode == 'spots':
            if ((0<=w<=5) and (0<=h<=5)):  
                for i in range(y,y+h):
                    for j in range(x, x+w):
                        new_img[i][j] = 255
        elif mode == 'linesVert':
            if h>100 and w<30:
                for i in range(y,y+h):
                    for j in range(x, x+w):
                        new_img[i][j] = 255
        elif mode == 'linesHoriz':
            if w>100 and h<30:
                for i in range(y,y+h):
                    for j in range(x, x+w):
                        new_img[i][j] = 255
        elif mode == 'linesBoth':
            if (w>100 and h<30) or (h>100 and w<30):
                for i in range(y,y+h):
                    for j in range(x, x+w):
                        new_img[i][j] = 255
                        
    return new_img

def printContours(binarized_img,output_img, thickness):
    '''
    Draw a green rectangle (if the image is not in grayscale) around to characters/words.
    
    Parameters
    ----------
    binarized_img : array of binarized image pixels.
    output_img: array of output image pixels. This will be modified at the end of the method.
    thickness: integer used to choose the thickness of the rectangle lines
    
    Returns
    -------
    box_coordinates: coordinates and dimension(x,y,w,h) of all bounding boxes that respect the condition.
    '''
    box_coordinates = [] 
    contours,_  = cv.findContours(~binarized_img,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE) 
    for contour in contours:
        [x,y,w,h] = cv.boundingRect(contour)
        if w >= 4 and h >= 4:
            cv.rectangle(output_img, (x,y), (x+w,y+h), (0, 255, 0), thickness)
        if (w >= 100 and h >= 50) or (w >= 50 and h >= 100):
            box_coordinates.append([x,y,w,h])
            
    return box_coordinates
            
def findMidDistanceContour(binarized_img, vertical: bool = False):
    '''
    Utility method of valueRLSA() to get the medium width of all CCs bounding boxes
    
    Parameters
    ----------
    binarized_img: array of image pixels binarized
    vertical: boolean value to enable the calculation the horizontal RLSA (default = False) (optional) 
    
    Returns
    -------
    mid_value: medium  width value of all CCs bounding boxes.
    '''
    contours,_  = cv.findContours(~binarized_img,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE) 
    sum_dir = 0
    size = 0
    for contour in contours:
        [x,y,w,h] = cv.boundingRect(contour)
        if vertical:
            sum_dir += h
        else:
            sum_dir += w
        size += 1
    mid_value = sum_dir/size
    return mid_value
    
def findDistance(binarized_img, vertical: bool = False):
    '''
    Utility method of valueRLSA() to find the distances between all CCs centroids
    
    Parameters
    ----------
    binarized_img: array of image pixels binarized
    vertical: boolean value to enable the calculation the horizontal RLSA (default = False) (optional) 
    
    Returns
    -------
    distances: array of distances between all CCs
    '''
    points = la.findCentroids(binarized_img,binarized_img.copy())
    G,edges = la.minimumSpanningTreeEdges(points,5)
    edges_dist = G.data
    edges_hor,edges_vert = edgesInformation(edges,points,edges_dist)
    distances = []
    edges = edges_hor
    if vertical:
        edges = edges_vert
    for edge in edges:
        c1,c2 = edge[2]
        x1,y1 = points[c1]
        x2,y2 = points[c2]
        dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        distances.append(dist)
       
    for dist in distances:
        if dist > 150:
            distances.remove(dist)
    
    return distances


def histogram(binarized_img, distance, vertical: bool = False):
    '''
    Method used to show the distance histogram of CCs centroids in the compute of adaptive RLSA.
    
    Parameters
    ----------
    binarized_img: array of image pixels binarized
    distance: array of distances between CCs centroids
    vertical: boolean value to enable the show of horizontal RLSA (default = False) (optional) 
    
    Returns
    -------
    value: integer of final rounded value to perform RLSA
    distances: array of distances between all CCs
    '''
    w=findMidDistanceContour(binarized_img,vertical)
    row_number = [i for i in range(len(distance))]
    distances=[]
    for j in range(len(distance)):
        distances.append(distance[j]-w)
    plt.bar(row_number,distances)
    plt.show()        
              
    
def rotate(original_img,binarized_img):
    '''
    Utility method used to rotate the image 
    
    Parameters
    ----------
    original_img: array of image pixels 
    binarized_img: array of image pixels binarized
    
    Returns
    -------
    rotated_img: array of image pixels after rotation
    '''
    img_no_figures = removeFiguresOrSpots(binarized_img, 'figures')
    rotated_img,_ = pp.houghTransformDeskew(img_no_figures, original_img, False)
    return rotated_img

def showProjection(binarized_img, counts, row_number):
    '''
    Utility method used to show the projection computed in the projection() method.
    
    Parameters
    ----------
    binarized_img: array of image pixels binarized
    counts: integer represents number of black pixels
    row_number: integer represent number of rows
    
    '''
    f, (ax1,ax2)= plt.subplots(1, 2, sharey=True, figsize=(70, 20))#(70,40)
    ax1.imshow(binarized_img,'gray')
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

def euclideanDistance(point1, point2):
    '''
    Calculates the euclidean distance between two points.
    
    Parameters
    ----------
    point1/2: pair of float coordinates x,y
    
    Returns
    -------
    distance: float distance value
    '''
    distance = (point2[0] - point1[0])**2 + (point2[1] - point1[1])**2
    return math.sqrt(distance)

def angleBetween(point1, point2):
    '''
    Calculates the inclination of the line passing through two point.
    
    Parameters
    ----------
    point1/2: pair of float coordinates x,y
    
    Returns
    -------
    degrees: float inclination value
    '''
    dx = point2[1]-point1[1]
    dy = point2[0]-point1[0]
    arctan = math.atan2(dx,dy)
    return math.degrees(arctan)

def getAngles(edges,points):
    '''
    Calculates the inclination of the line passing through two point for an array of edges.
    
    Parameters
    ----------
    edges: array of pair points edge index (tree edges)
    points: array of float pair coordinates x,y
    
    Returns
    -------
    angles: array of float angle values
    '''
    angles = []
    for edge in edges:
        i,j = edge
        angles.append(angleBetween(points[i],points[j]))
    return angles

def plotEdges(image, edges, points):
    '''
    Method that write edges on the image passed as parameter.
    
    Parameters
    ----------
    image: array of image pixels 
    edges: array of pair points edge index (tree edges)
    points: array of float pair coordinates x,y
    
    Returns
    -------
    output_img: array of image pixels with written edges
    '''
    points = np.int32(points)
    output_img = image.copy()
      
    for edge in edges:
        i,j = edge
        cv.line(output_img, (points[i, 0], points[i, 1]), (points[j, 0], points[j, 1]), (0,0,0), 1, cv.LINE_AA)
    return output_img

def edgesInformation(edges, points, distances):
    '''
    Utility method used for compute docstrum. It findes from a list of edges which are horizontal 
    and vertical usind a difference of +-20 degrees to remedy edge angle imprecision .
    
    Parameters
    ----------
    edges: array of pair points edge index (tree edges)
    points: array of float pair coordinates x,y
    distances: array of float values represents the distance between two points of a edge
    
    Returns
    -------
    horizontal_edges: array of pair points horizontal edge index 
    vertical_edges: array of pair points vertical edge index 
    '''
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


def counterClockwise(A,B,C):
    '''
    Utility method of intersect()
    '''
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])


def intersect(A,B,C,D):
    '''
    Method used to check intersection between two lines
    
    Parameters
    ----------
    A,B: points of the first line
    C,D: points of the second line
    
    Returns
    -------
    intersect: boolean value about intersection
    '''
    return counterClockwise(A,C,D) != counterClockwise(B,C,D) and counterClockwise(A,B,C) != counterClockwise(A,B,D)  


def findPeaks(distances, distance: int=0, plot: bool=False):
    '''
    Utility method used to find the two distances most present. 
    
    Parameters
    ----------
    distances: array of float values represents the distance between two points of a edge
    distance: integer value used as distance between the first and the second peak (optional)
    plot: boolean value used to enable peak histogram plot.
    Returns
    -------
    peak_values: integer values of best two peak distances
    '''
    d = defaultdict(int)
    
    for k in distances:
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
    
    sort_id = np.argsort(x)
    x = x[sort_id]
    y = y[sort_id]

    #findo optimum local points 
    if(distance!=0):
        peaks, _ = find_peaks(y, distance= distance)
    else:
        peaks, _ = find_peaks(y)


    peaks_occurrence_list=[]
    peaks_occurrences=[]
    for i in range(len(peaks)):
        peaks_occurrences.append(y[peaks[i]])
        peaks_occurrence_list.append((x[peaks[i]],y[peaks[i]]))
    peaks_occurrence_list = sorted(peaks_occurrence_list, key = lambda x:x[1], reverse=True)
    
    #finds the two higher peaks
    best_peaks=heapq.nlargest(2, peaks_occurrences)
    

    peak_values=[]
    for k in range(len(peaks_occurrence_list)):
        if len(peak_values)<2 and (best_peaks[0]==peaks_occurrence_list[k][1] or best_peaks[1]==peaks_occurrence_list[k][1]):
            peak_values.append(int(peaks_occurrence_list[k][0]))  
    #print('best peaks', best_peaks, 'my peak', peak_values)
    
    if plot:
        plt.plot(x, y)
        plt.plot(peak_values, best_peaks, "x")
        plt.xlim(0, 80)
        plt.show()
        print('Peak values for this image are',peak_values)
    return peak_values
