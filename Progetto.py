import cv2 as cv
import numpy as np
import math
import statistics
import os
import glob
import igraph
import networkx as nx
#from igraph import Graph, EdgeSeq
from matplotlib import pyplot as plt
from skimage.filters import (threshold_otsu, threshold_sauvola)
from skimage.transform import hough_line
from skimage import measure
import sklearn.preprocessing 
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.neighbors import kneighbors_graph


'''
Sistemare voronoi
ocr pytesseract
sbianchettare figure DONE
commentare e modulare semiDONE (da continuare insieme)

ho migliorato anche la creazione dei box in modo tale da evitare gli spots
'''

def binarization(mode,image):
    #AGGIUNGERE MOLT PER 255
    '''
    Method used for the binarization, using two different threshold (Otsu and Sauvola)
    
    Parameters
    ----------
    mode : string used for select the type of binarization
    image: array of image pixels. If it is not in grayscale, the image will be converted in grayscale.

    Returns
    -------
    binarization : array of image pixels binarized
    '''
    if image.ndim >2:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    if mode == 'otsu':
        return sklearn.preprocessing.binarize(image, threshold_otsu(image))
    elif mode == 'sauvola':
        return sklearn.preprocessing.binarize(image, threshold_sauvola(image))
    elif mode == 'inverse':
        _,thresh = cv.threshold(image,127,255,cv.THRESH_BINARY_INV)
        return thresh
        

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
    cv.imwrite(text_name+'.png', file_name)
    cv.waitKey(0)
    cv.destroyWindow(text_name)

def removeFiguresOrSpots(binarized_img,original_img, mode):
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
    if original_img.ndim >2:
        original_img = cv.cvtColor(original_img, cv.COLOR_BGR2GRAY)
    contours,_ = cv.findContours(np.uint8(np.logical_not(binarized_img)), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    new_img = original_img.copy()
    for contour in contours:
        [x,y,w,h] = cv.boundingRect(contour)
        if mode == 'figures':
            if w>400 or h>300 :#remove if the box it's too big (figure) or if it's too small (spot)
                for i in range(y,y+h):
                    for j in range(x, x+w):
                        new_img[i][j] = 255
        elif mode == 'spots':
            if ((0<=w<=7) and (0<=h<=7)):  #DACONTROLLARE (RIMUOVE PUNTI)
                for i in range(y,y+h):
                    for j in range(x, x+w):
                        new_img[i][j] = 255
    #showImage('gn',new_img)
    return new_img
        

def printContours(binarized_img,output_img):
    '''
    Draw a green rectangle (if the image is not in grayscale) around to characters/words.
    
    Parameters
    ----------
    binarized_img : array of binarized image pixels.
    output_img: array of output image pixels. This will be modified at the end of the method.
    '''
    
    contours,_  = cv.findContours(np.uint8(np.logical_not(binarized_img)),cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE) 
    for contour in contours:
        [x,y,w,h] = cv.boundingRect(contour)
        if w >= 4 and h >= 4:
            cv.rectangle(output_img, (x,y), (x+w,y+h), (0, 255, 0), 1)
 
    
def iteration(image, value: int):
    rows, cols = image.shape
    for row in range(rows):
        try:
            start = image[row].tolist().index(0)
        except ValueError:
            start = 0

        count = start
        for col in range(start, cols):
            if image[row, col] == 0:
                if (col-count) <= value and (col-count) > 0:
                    image[row, count:col] = 0               
                count = col  
    return image 


    #RLSA consiste nell'estrarre il blocco di testo o la Regione di interesse(ROI) dall'immagine binaria
    #del documento. Bisogna passargli un'immagine binaria di tipo ndarray.
def rlsa(image, horizontal: bool = True, vertical: bool = True, value: int = 0):
    if horizontal:
            image = iteration(image, value)
    if vertical: 
            image = image.T
            image = iteration(image, value)
            image = image.T
    return image
   
def showCC(binarized_img,conn):
    
    cc, labels = cv.connectedComponents(~binarized_img,connectivity=conn)
    
    
    label_color = np.uint8(179*labels/(cc-1))
    blank_channel = 255*np.ones_like(label_color)#creates a white matrix
    labeled_img = cv.merge([label_color, blank_channel, blank_channel])#creates an image in RGB
    
    labeled_img = cv.cvtColor(labeled_img, cv.COLOR_HSV2BGR)#convert to BGR for display
    
    labeled_img[label_color==0] = 0 #set backgroud label to black
    
    if conn==4:
        print('number of CC with connectivity 4:',cc)
        titles = ['Original Image','CC with connettivity 4']
    elif conn==8:
        print('number of CC with connectivity 8:',cc)
        titles = ['Original Image','CC with connettivity 8']
    images = [binarized_img, labeled_img]
    for i in range(2):
        plt.figure(figsize=(20,20))
        plt.subplot(1,2,i+1)
        plt.imshow((images[i]),'gray')
        plt.title(titles[i])
        plt.axis('off')

    plt.show()
    


def findMidDistanceContour(binarized_img):
    contours,_  = cv.findContours(np.uint8(np.logical_not(binarized_img)),cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE) 
    sumW = 0
    size = 0
    for contour in contours:
        [x,y,w,h] = cv.boundingRect(contour)
        sumW += w
        size += 1
    midW = sumW/size
    return midW
    

def valueRLSA(binarized_img, vert: bool = False):
    
    distance = np.asarray(findDistance(binarized_img, vert))
    numSum=0
    sumDist=0 
    distance = distance.astype(np.int)
    
    for k in range(distance.shape[0]):
        sumDist += distance[k]
        numSum += 1  
           
    value=sumDist/numSum
    midW = findMidDistanceContour(binarized_img)
    value -= midW
    return round(value),distance

#trova la distanza tra centroidi
def findDistance(binarized_img, vert):
    points = findCentroids(binarized_img,binarized_img.copy())
    G,edges = minimumSpanningTreeEdges(points,4)
    edges_dist = G.data
    edges_hor,edges_vert = edgesInformation(edges,points,edges_dist)
    distance = []
    edges = edges_hor
    if vert:
        edges = edges_vert
    for edge in edges:
        c1,c2 = edge[2]
        x1,y1 = points[c1]
        x2,y2 = points[c2]
        dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        distance.append(dist)
    return distance


def histogram(binarized_img,distance):
    w=findMidDistanceContour(binarized_img)
    row_number = [i for i in range(len(distance))]
    distances=[]
    for j in range(len(distance)):
        distances.append(distance[j]-w)
    plt.bar(row_number,distances)
    plt.show()        
    


   

def houghTransformDeskew(binarized_img,original_img, plot : bool = True):
    '''
    Compute deskew angle using Hough transform, it also plot the histogram of Hough
    transform and build the deskew of the image.
    
    Parameters
    ----------
    binarized_img : array of binarized image pixels.
    original_img: array of original image pixels.

    Returns
    -------
    img_rotated : array of deskewed image pixels, or None if the image is not skewed.
    '''
    
    edges = cv.Canny(binarized_img, 50, 200, 3)#find edges on the image
    img_lines = cv.cvtColor(edges, cv.COLOR_GRAY2BGR) #convert edges image from Gray to BGR

    lines = cv.HoughLinesP(edges, 1, np.pi/180, 80, None, 100, 10) #function used for finding coordinates x0,y0 and x1,y1 for deskew
    tested_angles = np.linspace(-np.pi/2, np.pi / 2, 360)
    h, theta, d = hough_line(edges,tested_angles)#function used for plot histogram Hough transform
    
    if lines is not None:
        angle = 0.0
        num_lines = len(lines)
        
        for i in range(num_lines):
            #write blue lines on image according to Hough lines
            l = lines[i][0]
            cv.line(img_lines, (l[0], l[1]), (l[2], l[3]), (255,0,0), 3, cv.LINE_AA)
            angle += math.atan2(l[3]*1.0 - l[1]*1.0,l[2]*1.0 - l[0]*1.0)
           
        angle /= num_lines #averages between all found angles
        best_angle = angle* 180.0 / np.pi #find the best angle in the right notation
        
        print(best_angle) 
        #showImage('Detected Lines with Probabilistic Line Transform', img_lines)#non so se puo essere utile ai fini del progetto stampare le linee 
        
        (height, width) = original_img.shape[:2]
        center = (width // 2, height // 2)
        if plot:
            #show histogram of Hough transform
            plt.figure(figsize=(20,20))
            plt.imshow(np.log(1 + h), extent=[np.rad2deg(theta[-1]), np.rad2deg(theta[0]), d[-1], d[0]], cmap ='nipy_spectral', aspect=1.0 / (height/30))
            plt.title('Histogram Hough Transform')
            plt.show()
        #build the deskew
        root_mat = cv.getRotationMatrix2D(center, best_angle, 1)
        img_rotated = cv.warpAffine(original_img, root_mat, (width,height), flags=cv.INTER_CUBIC, borderMode=cv.BORDER_REPLICATE)
        img_rotated_no_fig = cv.warpAffine(binarized_img, root_mat, (width,height), flags=cv.INTER_CUBIC, borderMode=cv.BORDER_REPLICATE) 
        return img_rotated, img_rotated_no_fig
    return None

def rotate(img_orig,img_bin):
    img_copy = img_orig.copy()
    printContours(img_bin,img_copy)       

    valueH,_= valueRLSA(img_bin)
    valueV,_ = valueRLSA(img_bin,True)
    
    img_rlsa_H = rlsa(img_bin.copy(), True, False, valueH)
    img_rlsa_full = rlsa(img_rlsa_H.copy(),False,True,valueV)
    
    img_no_figures = removeFiguresOrSpots(img_rlsa_full, img_orig, 'figures')
    img_rotated,_ = houghTransformDeskew(img_no_figures, img_orig, False)
    return img_rotated
    

def projection(img_bin):
    
    #Counting black pixels (img_bin==0) per row (axis=0: col, axis=1:row)
    counts = np.sum(img_bin==0, axis=1) 
    row_number = [i for i in range(img_bin.shape[0])]
    #counts = smooth(counts,20) #ammorbidisce il grafico dei pixel
    return counts, row_number
  
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
    
   


def findCentroids(binarized_img,output_img):
    '''
    Find centroids of connected components (CCs) and draws it on the image passed as parameter.
    
    Parameters
    ----------
    binarized_img : array of binarized image pixels.
    output_img: array of output image pixels. This will be modified at the end of the method.

    Returns
    -------
    points : array of coordinates (x,y) about the finded centroids, sorted in increasing order. 
    '''
    contours,_  = cv.findContours(np.uint8(np.logical_not(binarized_img)),cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE) 
    points = []
    for contour in contours:
        M = cv.moments(contour)
        # calculate x,y coordinate of center
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            points.append((cX,cY))
        else:
            cX, cY = 0, 0
            
        cv.circle(output_img, (cX, cY), 5, (0, 255, 0), -1)
    points.sort(key=lambda x:x[0])
        
    return points

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

def drawDelaunay(img, subdiv, delaunay_color) :
    '''
    Draw the Delaunay triangulation using Delaunay subdivision.
    
    Parameters
    ----------
    img: array of image pixels.
    subdiv : Delaunay subdivision of points (centroids)
    delaunay_color: color in RGB for lines draw

    '''
    triangleList = subdiv.getTriangleList() #gets the triangle list from Delaunay subdivision
    size = img.shape
    rect = (0, 0, size[1], size[0])

    for t in triangleList:
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])
        if rectContains(rect, pt1) and rectContains(rect, pt2) and rectContains(rect, pt3) :
            cv.line(img, pt1, pt2, delaunay_color, 1, cv.LINE_AA, 0)
            cv.line(img, pt2, pt3, delaunay_color, 1, cv.LINE_AA, 0)
            cv.line(img, pt3, pt1, delaunay_color, 1, cv.LINE_AA, 0)

def drawVoronoi(img, subdiv) :
    '''
    Draw the Voronoi Area diagram using Delaunay subdivision.
    
    Parameters
    ----------
    img: array of image pixels.
    subdiv : Delaunay subdivision of points (centroids)

    '''
    facets, _ = subdiv.getVoronoiFacetList([]) #get the Voronoi facet list 
    #img2 = img.copy()
    for i in range(len(facets)) :
        ifacet_arr = []
        for f in facets[i] : #takes the numpy array of facet points and puts it in a new array
            ifacet_arr.append(f)
        ifacet = np.array(ifacet_arr, np.int) #makes the numpy array in a simple integer array
        ifacets = np.array([ifacet])#builds an array of the previous integer array for make it compatible with the following method.
        height,width = img.shape[:2]
        for i in range(len(ifacets[0])-1):
            x1,y1 = ifacets[0][i]
            x2,y2 = ifacets[0][i+1]
        
            if x1>=0 and x2>=0 and y1>=0 and y2>=0:
                if x1<=width and x2<=width and y1<=height and y2<=height: 
                    #print(euclidean_distance(ifacets[0][i],ifacets[0][i+1]))
                    #if(euclidean_distance(ifacets[0][i],ifacets[0][i+1])<70):
                    cv.line(img, (x1, y1), (x2, y2), (0,0,255), 1, cv.LINE_AA)
                
    
        #cv.polylines(img2, ifacets, True, (255, 0, 0), 1, cv.LINE_AA, 0)#draws facet lines from the array of array ifacets
    #showImage('met 1',img)
    #showImage('met 2',img2)

def voronoi(points,img):
    '''
    Build Delaunay Triangulation and Voronoi Area 
    
    Parameters
    ----------
    points : array of coordinates (x,y) about the centroids.
    img: array of image pixels.
    '''
    img_voronoi = img.copy()
    size = img.shape
    rect = (0, 0, size[1], size[0]) #creates a tuple with the origin and size of the image rectangle
    subdiv = cv.Subdiv2D(rect) #creates an empty Delaunay subdivision with the rectangle size
    
    for p in points : #add 2D points into Delaunay subdivisiom
        subdiv.insert(p)
        
    
    drawDelaunay(img, subdiv, (0, 0, 255)) #draw the triangulation using Delaunay subdivision.
    for p in points :
        cv.circle(img, p, 2, (255,0,0), cv.FILLED, cv.LINE_AA, 0 )
    drawVoronoi(img_voronoi,subdiv) #draw the Voronoi diagram using Delaunay subdivision.
    #showImage('Delaunay Triangulation',image)
    showImage('Voronoi Diagram',img_voronoi)
    

def kNeighborsGraph(points, k):
    '''
    Builds a K-Neighbors graph from an list of coordinates (x, y)
    
    Parameters
    ----------
    points : array of coordinates (x,y).
    k : int the number of neighbor to consider for each vector.
    
    Returns:
    G : sparse K-Neighbors graph     
    '''
    
    # k = len(X)-1 gives the exact MST
    k = min(len(points) - 1, k)
    
    # generate a sparse graph using the k nearest neighbors of each point
    return kneighbors_graph(points, n_neighbors=k, mode='distance')

def minimumSpanningTreeEdges(points, k): 
    '''
    Builds a Minimum Spanning Tree from an list of coordinates (x, y)
    
    Parameters
    ----------
    points : array of coordinates (x,y).
    k : int the number of neighbor to consider for each vector.
    
    Returns:
    full_tree :
    edges : numpy array of connected nodes in the MST. Index couple of points.     
    '''
    G = kNeighborsGraph(points, k)
    
    # Compute the minimum spanning tree of this graph
    full_tree = minimum_spanning_tree(G, overwrite=True)

    return full_tree,np.array(full_tree.nonzero()).T 


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
    plt.figure(figsize=(30,20))
    plt.imshow(img, 'gray')  
    for edge in edges:
        #i, j = edge
        i,j = edge
        plt.plot([points[i, 0], points[j, 0]], [points[i, 1], points[j, 1]], c='r')
    plt.show() 
    

def docstrum(input_img, output_img, edges, points, thresh_dist):
    plt.figure(figsize=(30,20))
    plt.imshow(input_img, 'gray')    
    for edge in edges:
        i,j = edge[2]
        distan = edge[1]
        
        if distan < thresh_dist:
            cv.line(output_img, (points[i, 0], points[i, 1]), (points[j, 0], points[j, 1]), (0,0,0), 3, cv.LINE_AA)
            plt.plot([points[i, 0], points[j, 0]], [points[i, 1], points[j, 1]], c='r')
    plt.show()

def edgesInformation(edges, points, distances):
    angles = getAngles(edges,points)
    

    horizontal_edges =[]
    vertical_edges = []
    munnezza = []
    for i in range(len(angles)):
    
        #if -24< angles[i] < 24 or 156 < angles[i] or angles[i] < -156:
        if -15< angles[i] < 15 or 165 < angles[i] or angles[i] < -165:
            horizontal_edges.append((angles[i],distances[i],[edges[i][0],edges[i][1]]))
        elif 75 < angles[i] < 105 or (-75 > angles[i] and angles[i] > -105)  :
            vertical_edges.append((angles[i],distances[i],[edges[i][0],edges[i][1]]))
        else:
            munnezza.append((angles[i],distances[i],[edges[i][0],edges[i][1]]))
    '''
    print('HORIZ')
    for i in range(len(horizontal_edges)):
        print(horizontal_edges[i])
    print('VERT')
    for i in range(len(vertical_edges)):
        print(vertical_edges[i])
    print('MUN')
    for i in range(len(munnezza)):
        print(munnezza[i])
    '''
    return horizontal_edges,vertical_edges


def cutImage(image_bin,nPixel,space,verticalCut: bool = False):
    if verticalCut:
        image_bin = image_bin.T
        
    #Counting black pixels per row (axis=0: col, axis=1:row)
    counts,_ = projection(image_bin)

    #cut contiene tutte le righe che hanno meno di nPixel pixel
    cut=[]
    for i in range(counts.shape[0]):
        if(counts[i]<nPixel):
            cut.append(i)
    x=0
    h=0
    info=[]
    flag=False
    for j in range(len(cut)-1):
        if cut[j+1]-cut[j]==1:
            if flag==False:
                h=cut[j]
                flag=True
            x=x+1
        else:
            info.append([x,h,cut[j]])
            flag=False
            x=0
    info.append([x,h,cut[j]])
    
    delete=[]
    for k in range(len(info)):
        if info[k][0]<space:
            delete.append(k)
    for m in range(len(delete)-1,-1,-1):
        info.remove(info[delete[m]])
    #print('[[nPixel,pStart, pEnd]]:',info)
    
    
    if verticalCut:
        image_bin = image_bin.T
        cv.imwrite('verticalCut.tif', image_bin)
    else:
        cv.imwrite('horizontalCut.tif', image_bin)
    return info

def cutMatrix(img_name, path, img_bin, info, infoV, XY_Tree):
    #img_name='prova30.tif'
    #img = cv.imread(img_name, 0)
    imgV = img_bin# cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    
    #path = 'Users\\manus\\Desktop\\Progetto DDM\\XY_Tree'
    
    #horizontal cut
    #imgV = cv2.imread(newName, 0)
    for i in range(len(info)-1):
        crop_img = imgV[info[i][2]:info[i+1][1],0:imgV.shape[1]]
        new_imgname=img_name[:-4]+'_horizCrop'+str(i)+'.tif'
        cv.imwrite(os.path.join(path , new_imgname),crop_img)
        #cv2.imwrite(new_imgname, crop_img)
        #plt.imshow(crop_img, 'gray'),plt.show()
    #tree = Node()
    #tree.name = 'Page(root)'

    typeNode =[]
    label=[]
    
    XY_Tree.add_vertices(1)
    label.append('Pg')
    typeNode.append('root')
    
    XY_Tree_Node=0
    
    
    #vertical cut
    filelist = []
    for infile in glob.glob (os.path.join (path, 'prova*_horizCrop*.tif')):
        filelist.append (infile)
    filelist.sort()
    #x=0
    #y=0
    
    for file in filelist:
        newRead = cv.imread(file, 0)
        inf = cutImage(newRead,4,10,True)#4,21
        if len(inf)==1:
            print('NoCut')
            title=input("Name: ")
            #tree.add_child()
            #tree[x].name = title
            plt.title(title)
            plt.imshow(newRead, 'gray'),plt.axis('off'),plt.show()
        else:
            flag=False
            for h in range(len(inf)-1):
                cropV = newRead[ 0:newRead.shape[0], inf[h][2]:inf[h+1][1] ]
                newNameV = file[:-4]+'_FinalVC'+str(h)+'.tif'
                cv.imwrite(os.path.join(path , newNameV), cropV)
                title=input("Name: ")
                XY_Tree_Node += 1
                if len(inf)<=2:
                    XY_Tree.add_vertices(1)
                    label.append(title)
                    typeNode.append('leaf')
                    XY_Tree.add_edges ([(0,XY_Tree_Node)])
                    #tree.add_child()
                    #tree[x].name = title
                    #x=x+1
                elif flag==False:
                    #tree.add_child()
                    #tree[x].name = 'O'
                    #tree[x].add_child()
                    #tree[x][y].name = title
                    XY_Tree.add_vertices(1)
                    label.append(' ')
                    typeNode.append('node')
                    XY_Tree.add_edges ([(0,XY_Tree_Node)])
                    XY_Tree.add_vertices(1)
                    label.append(title)
                    typeNode.append('leaf')
                    count = 0
                    XY_Tree.add_edges ([(XY_Tree_Node,XY_Tree_Node+1)])
                    #x=x+1
                    flag=True
                else:
                    #tree[x-1].add_child()
                    #tree[x-1][y].name = title
                    XY_Tree.add_vertices(1)
                    label.append(title)
                    typeNode.append('leaf')
                    count += 1
                    XY_Tree_Node += 1
                    XY_Tree.add_edges ([(XY_Tree_Node-2*count,XY_Tree_Node-count+1)])
                    
                #y=y+1
                #flag=False
                plt.title(title)
                plt.imshow(cropV, 'gray'),plt.axis('off'), plt.show()
        #y=0
    
    #print(tree)
    print(XY_Tree)
    #return pippo,typeNode,label
    return typeNode,label

def fanculo():
    '''
    g = igraph.Graph()
    g.add_vertices(5)
    g.add_edges ([(0,1),(0,2),(0,3),(0,4)])
    g.add_vertices(1)
    g.add_edges ([(4,5)])
    g.add_vertices(1)
    g.add_edges ([(4,6)])

    g.vs["name"] = ["A", "B", "C", "D","E","F","G"]
    g.vs["type"] = ["root", "leaf", "leaf", "leaf","node","leaf","leaf"]
    
    g.vs["label"] = g.vs["name"]
    color_dict = {"root": "white", "leaf": "red", "node": "yellow"}
    g.vs["color"] = [color_dict[gender] for gender in g.vs["type"]]

    layout = g.layout("tree")
    igraph.plot(g, layout = layout, bbox = (200, 200), margin = 20)
    '''
    '''
    g = nx.DiGraph()
    g.add_node(5)
    g.add_edges_from([(0,1),(0,2),(0,3),(0,4)])
    g.add_node(1)
    g.add_edges_from([(4,5)])
    g.add_node(1)
    g.add_edges_from([(4,6)])
    '''
    '''
    G = nx.DiGraph()
    G.add_node("ROOT")
    for i in range(5):
        G.add_node("Child_%i" % i)
        G.add_node("Grandchild_%i" % i)
        G.add_node("Greatgrandchild_%i" % i)

        G.add_edge("ROOT", "Child_%i" % i)
        G.add_edge("Child_%i" % i, "Grandchild_%i" % i)
        G.add_edge("Grandchild_%i" % i, "Greatgrandchild_%i" % i)
    #nx.draw(G, with_labels=True, font_weight='bold')
    
    plt.title('draw_networkx')
    nx.graphv
    pos=nx.graphviz_layout(G, prog='dot')
    nx.draw(G, pos, with_labels=False, arrows=False)
    plt.savefig('nx_test.png')
    
    #nx.draw_shell(g, nlist=[range(5, 10), range(5)], with_labels=True, font_weight='bold')
    '''
    G = nx.complete_graph(5)   # start with K5 in networkx
    A = nx.nx_agraph.to_agraph(G)        # convert to a graphviz graph
    X1 = nx.nx_agraph.from_agraph(A)     # convert back to networkx (but as Graph)
    X2 = nx.Graph(A)          # fancy way to do conversion
    G1 = nx.Graph(X1)          # now make it a Graph
    nx.draw(G1, with_labels=True, font_weight='bold')


def main():
    
    #img = cv.imread("Immagini prova\\N0024670aau.tif")  #Leggo l'immagine 
    img = cv.imread("skew1.png")
    img_word = img.copy()
    img_centroids = img.copy()
    SHOWSTEPS = True
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    bin_otsu = binarization('otsu',img_gray)
    bin_sauvola = binarization('sauvola',img_gray)
    bin_inv = binarization('inverse',img_gray)
    
    if SHOWSTEPS:
        
    
        showImage('Original Image',img)
        showImage('Otsu Binarization',bin_inv)
        showImage('Otsu Binarization',bin_otsu*255)
        showImage('Sauvola Binarization', bin_sauvola*255)

    #bin_sauvola = cv.medianBlur(bin_sauvola, 3) #apply median blur to remove black spots on images
    #showImage('Sauvola Binarization BLUR', bin_sauvola*255)
    
    rot = rotate(img,bin_sauvola)
    showImage('rot',rot)
    
    
    
    img_no_spots = removeFiguresOrSpots(bin_sauvola,img,'spots')
    bin_no_spots = binarization('sauvola',img_no_spots)
    
    showCC(bin_no_spots*255, 8)
    
    
    
    img = img_no_spots
    bin_sauvola = bin_no_spots
    
    img_bin_char = bin_sauvola.copy()
    printContours(img_bin_char,img_bin_char)    
    showImage('Characters Contour', img_bin_char*255)    

    valueH,dist = valueRLSA(bin_sauvola)
    histogram(bin_sauvola,dist)
    print(valueH)
    valueV, dist = valueRLSA(bin_sauvola,True)
    histogram(bin_sauvola,dist)
    print(valueV)
    
    img_rlsa_H = rlsa(bin_sauvola.copy(), True, False, valueH)
    img_rlsa_full = rlsa(img_rlsa_H.copy(),False,True,valueV)

    showImage('RLSA',img_rlsa_full*255)
    printContours(img_rlsa_full,img_word)
    showImage('Words Contour', img_word)
    
    img_no_figures = removeFiguresOrSpots(img_rlsa_full,img,'figures')
    showImage('Image Without Figures', img_no_figures)
 
    img_rotated, img_rotated_no_fig = houghTransformDeskew(img_no_figures,img)

    if img_rotated is not None:
        #showImage('Rotated Image', img_rotated)
        bin_rotated = binarization('sauvola',img_rotated)
        img = img_rotated_no_fig.copy()
    else:
        print('Image not skewed')

    img_voro = img.copy() 
    img_k = img.copy()
    img_docstrum_lines = img.copy()
    img_docstrum_box = img.copy()
    
    #projection(bin_rotated,rotated)
    
    points = findCentroids(bin_rotated,img_centroids)
    showImage('Centroids',img_centroids)
    voronoi(points, img_voro) 
    
    Graph = kNeighborsGraph(points,5)           
    k_kneighbors_edges = np.array(Graph.nonzero()).T
    k_kneighbors_distances = Graph.data
    
    media_dist = statistics.mean(k_kneighbors_distances)
    print('media',media_dist)
    print('max',max(k_kneighbors_distances))
    print('min',min(k_kneighbors_distances))

    row_number = [i for i in range(k_kneighbors_distances.shape[0])]
    plt.figure(figsize=(20,10))
    plt.bar(row_number,k_kneighbors_distances)
    plt.show()


    _,mst_edges= minimumSpanningTreeEdges(points,5)
    
    
    points = np.int32(points)

    
    plotEdges(img,k_kneighbors_edges,points)
    plotEdges(img,mst_edges,points)
    
    horizontal_edges, vertical_edges = edgesInformation(k_kneighbors_edges, points, k_kneighbors_distances)
    '''
    angles = getAngles(k_kneighbors_edges,points)
    
    horizontal_edges =[]
    vertical_edges = []
    for i in range(len(angles)):
    
        if -24< angles[i] < 24 or 156 < angles[i] or angles[i] < -156:
        #if -10< angles[i] < 10 or 170 < angles[i] or angles[i] < -170:
            horizontal_edges.append((angles[i],k_kneighbors_distances[i],[k_kneighbors_edges[i][0],k_kneighbors_edges[i][1]]))
        else:
            vertical_edges.append((angles[i],k_kneighbors_distances[i],[k_kneighbors_edges[i][0],k_kneighbors_edges[i][1]]))
            '''
    
    #DOCSTRUM ORIZZONTALE
    docstrum(img, img_k, horizontal_edges, points, max(k_kneighbors_distances)/2)
    
    img_k_orig = img_k.copy()

    showImage('K-NN Lines',img_k) 
    bin_img_k = binarization('sauvola',img_k)
    rlsa_docstrum = rlsa(bin_img_k.copy(), True, False, 2)
    showImage('K-NN Lines rlsa',rlsa_docstrum*255) 
    printContours(rlsa_docstrum,img_docstrum_lines)
    showImage('Docstrum lines',img_docstrum_lines)
    
    

    #DOCSTRUM VERTICALE
    docstrum(img, img_k, vertical_edges, points, max(k_kneighbors_distances)/2)
    


    showImage('K-NN Lines',img_k) 
    bin_img_k = binarization('sauvola',img_k)
    rlsa_docstrum = rlsa(bin_img_k.copy(), False, True, 1)
    printContours(rlsa_docstrum,img_docstrum_box)
    showImage('Docstrum lines',img_docstrum_box)
    
def main1():
    fanculo()
    img_name = 'Prova0.tif'
    img = cv.imread(img_name)
    img_bin = binarization('otsu', img)
    info=cutImage(img_bin,50,40)
    infoV=cutImage(img_bin,50,20,True)
    XY_Tree = igraph.Graph()
    #pippo, typeNode, label=cutMatrix(img_name, img_bin, info, infoV, pippo)
    typeNode, label=cutMatrix(img_name, img_bin, info, infoV, XY_Tree)
    XY_Tree.vs["type"] = typeNode
    XY_Tree.vs["label"] = label
    color_dict = {"root": "white", "leaf": "red", "node": "yellow"}
    XY_Tree.vs["color"] = [color_dict[type] for type in XY_Tree.vs["type"]]

    print(XY_Tree)
    #layout = pippo.layout("tree")
    layout = XY_Tree.layout_reingold_tilford(mode="in", root=0)
    igraph.plot(XY_Tree, layout = layout, bbox = (200, 200), margin = 20)
    


if __name__ == "__main__":
    main()


    





