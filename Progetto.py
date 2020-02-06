import cv2 as cv
import numpy as np
import math
from matplotlib import pyplot as plt
from skimage.filters import (threshold_otsu, threshold_sauvola)
from skimage.transform import hough_line,hough_line_peaks
from skimage import measure
from pythonRLSA import rlsa
import sklearn.preprocessing 
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.neighbors import kneighbors_graph


#img = cv.imread("N0024670aao.tif")  #Leggo l'immagine 
img = cv.imread("K.png")
img_word = img.copy()
img_centroids = img.copy()
img_voro = img.copy() 
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

def binarization(mode,image):
    if image.ndim >2:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    if mode == 'otsu':
        return sklearn.preprocessing.binarize(image, threshold_otsu(image))
    elif mode == 'sauvola':
        return sklearn.preprocessing.binarize(image, threshold_sauvola(image))
        
bin_otsu = binarization('otsu',img_gray)
bin_sauvola = binarization('sauvola',img_gray)

'''
plt.figure(figsize=(50,45))
plt.subplot(3, 2, 1)
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
#plt.imshow(img, 'gray')
plt.title('Original')
plt.axis('off')

plt.subplot(3, 2, 2)
plt.imshow(bin_otsu, 'gray')
plt.title('Otzu')
plt.axis('off')

plt.subplot(3, 2, 3)
plt.imshow(bin_sauvola, 'gray')
plt.title('Sauvola')
plt.axis('off')
'''

def showImage(text_name, file_name):
    '''
    cv.namedWindow(text_name, cv.WINDOW_NORMAL)
    cv.imshow(text_name, file_name)
    cv.imwrite(text_name+'.png', file_name)
    cv.waitKey(0)
    cv.destroyWindow(text_name)
    '''
    
showImage('Original Image',img)
showImage('Otsu Binarization',bin_otsu*255)
showImage('Sauvola Binarization', bin_sauvola*255)

bin_sauvola = cv.medianBlur(bin_sauvola, 3) #apply median blur to remove black spots on images
showImage('Sauvola Binarization BLUR', bin_sauvola*255)

labels = measure.label(bin_sauvola, background = 1, connectivity=2)

plt.figure(figsize=(20,20))
plt.imshow(labels, 'nipy_spectral')
plt.title('Connected Components')
plt.axis('off')
plt.show()

def removeFigures(binarized_img,original_img):
    contours,_ = cv.findContours(np.uint8(np.logical_not(binarized_img)), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    mask = np.ones(original_img.shape[:2], dtype="uint8") * 255
    for contour in contours:
        [x,y,w,h] = cv.boundingRect(contour)
        if w>300 or h>300:
            cv.drawContours(mask, [contour], -1, 0, -1)
    original_img = cv.bitwise_and(~original_img, ~original_img, mask=mask)
    return ~original_img
        

def printContours(binarization,output_img):
    #draw a green rectangle around to characters/words
    contours,_  = cv.findContours(np.uint8(np.logical_not(binarization)),cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE) 
    
    for contour in contours:
        [x,y,w,h] = cv.boundingRect(contour)
        cv.rectangle(output_img, (x,y), (x+w,y+h), (0, 255, 0), 1)
 
   
img_bin_char = bin_sauvola.copy()
printContours(img_bin_char,img_bin_char)    
showImage('Characters Contour', img_bin_char*255)    



def pixelDistance(image, vertical: bool = False):
    if vertical:
        image = image.T #la uso per calcolare la distanza tra pixel in verticale
    rows, cols = image.shape
    distance=[]
    count=0
    i1=0
    flag=0
    #i2=0

    for i in range(rows):
        for j in range(cols):
            
            if image[i][j]==255 and image[i][0]==255:
                '''
                if i!=i2:
                    flag=1
                    i2=i
                else:
                    flag=flag+1
                    i2=i
                    '''
            if image[i][j]==0:
                if i!=i1:
                    distance.append('-1')
                    i1=i
                    count=0
                elif j-count-flag!=0:
                        distance.append(j-count-flag)
                        flag=0
                        count=j
                    
    distance=np.array(distance)
    return distance

def valueRLSA(distance):
    rows=distance.shape[0]
    numSum=0
    sumDist=0 
    distance = distance.astype(np.int)
    
    for k in range(rows):
        if distance[k]!= -1:
            sumDist=sumDist+distance[k]
            numSum=numSum+1  
            
    value=sumDist/numSum
    print(value)
    return value

def histogram(image,distance):
    rows=distance.shape[0]
    distance = distance.astype(np.int)
    max=0
    for h in range(rows):
        if distance[h]>max:
            max=distance[h]
    
    #serve per rendere il grafico piu leggibile, elimina pixel a distanza 1
    q1=0
    for l in range(rows):
        if distance[l]==1 and q1<200:
            q1=q1+1
        elif distance[l]==1:
            distance[l]=-1
            
    #plt.imshow(image,'gray'), plt.xticks([]), plt.yticks([]), plt.show()
    plt.hist(distance,256,[0,30]), plt.show()
    #plt.savefig('hist.png', dpi=1800)
    

'''
pix_dist_horiz=pixelDistance(bin_sauvola*255)
pix_dist_vert=pixelDistance(bin_sauvola*255,True)
value_horiz=valueRLSA(pix_dist_horiz)
value_vert=valueRLSA(pix_dist_vert)
histogram(bin_sauvola,pix_dist_horiz)
histogram(bin_sauvola,pix_dist_vert)
'''
#DA VERIFICARE RLSA ADATTIVO, HO FATTO VARI TEST MA LA DIVISIONE DELLE PAROLE NON FUNZIONA BENE

#img_rlsa_oriz = rlsa.rlsa(img_bin_char.copy(), True, False, 10)  
#img_rlsa_full = rlsa.rlsa(img_rlsa_oriz.copy(), False, True, 3)

img_rlsa_full = rlsa.rlsa(bin_sauvola.copy(), True, True, 10)

showImage('RLSA',img_rlsa_full*255)
printContours(img_rlsa_full,img_word)
showImage('Words Contour', img_word)





   

def houghTransformDeskew(binarized_img,original_img):
  
    edges = cv.Canny(binarized_img, 50, 200, 3)#find edges on the image
    img_lines = cv.cvtColor(edges, cv.COLOR_GRAY2BGR) #convert edges image from Gray to BGR

    lines = cv.HoughLinesP(edges, 1, np.pi/180, 80,None,100,10) #function used for finding coordinates x0,y0 and x1,y1 for deskew
    tested_angles = np.linspace(-np.pi/2, np.pi / 2, 360)
    h, theta, d = hough_line(edges,tested_angles)#function used for plot histogram Hough transform
    
    if lines is not None:
        angle = 0.0
        num_lines = len(lines)
        
        for i in range(0, len(lines)):
            #write blue lines on image according to Hough lines
            l = lines[i][0]
            cv.line(img_lines, (l[0], l[1]), (l[2], l[3]), (255,0,0), 3, cv.LINE_AA)
            angle += math.atan2(l[3]*1.0 - l[1]*1.0,l[2]*1.0 - l[0]*1.0)
           
        
        angle /= num_lines*1.0
        best_angle = angle* 180.0 / np.pi
        
        print(best_angle) 
        showImage('Detected Lines with Probabilistic Line Transform', img_lines)#non so se puo essere utile ai fini del progetto stampare le linee 
        
        (height, width) = original_img.shape[:2]
        center = (width // 2, height // 2)
        print(height)
        #show histogram
        plt.figure(figsize=(20,20))
        plt.imshow(np.log(1 + h), extent=[np.rad2deg(theta[-1]), np.rad2deg(theta[0]), d[-1], d[0]],cmap ='nipy_spectral',aspect=1.0 / (height/30))
        plt.title('Histogram Hough Transform')
        plt.show()
        root_mat = cv.getRotationMatrix2D(center, best_angle, 1)
        rotated = cv.warpAffine(original_img, root_mat, (width,height), flags=cv.INTER_CUBIC,borderMode=cv.BORDER_REPLICATE) 
        return rotated
    return None

img_no_figures = removeFigures(img_rlsa_full,img)
showImage('Image Without Figures', img_no_figures)
 
rotated = houghTransformDeskew(img_no_figures,img)

if rotated is not None:
    showImage('Rotated Image', rotated)
    bin_rotated = binarization('sauvola',rotated)
else:
    print('Image not skewed')


def projection(img_bin, img_gray):
    #Counting black pixels per row (axis=0: col, axis=1:row)
    counts = np.sum(img_bin==0, axis=1) 
    row_number = [i for i in range(img_bin.shape[0])]
    #counts = smooth(counts,20) #ammorbidisce il grafico dei pixel
    
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(30, 15))
    ax1.imshow(img_gray,'gray'), plt.yticks([])
    ax1.tick_params(axis='both', which='major', labelsize=10)
    ax2.plot(counts,row_number,label='fit')
    ax2.tick_params(axis='both', which='major', labelsize=10)
    plt.xlabel('Number of Black Pixels',fontsize=20)
    plt.ylabel('Row Number',fontsize=20)
    plt.subplots_adjust( wspace = -0.1)
    plt.savefig('Projection.png')#, dpi=1800)
    
   
#projection(bin_rotated,rotated)

def findCentroids(binarization,output_img):
    contours,_  = cv.findContours(np.uint8(np.logical_not(binarization)),cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE) 
    points = []
    f = open('punti.txt', 'w')
    
    for contour in contours:
        M = cv.moments(contour)
 
        # calculate x,y coordinate of center
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            points.append((cX,cY))
            f.write(str(cX) + ' ' + str(cY) + '\n')
        else:
            cX, cY = 0, 0
            
        cv.circle(output_img, (cX, cY), 5, (0, 255, 0), -1)
        points.sort(key=lambda x:x[0])
    f.close()
    return points

def rect_contains(rect, point) :
    if point[0] < rect[0] :
        return False
    elif point[1] < rect[1] :
        return False
    elif point[0] > rect[2] :
        return False
    elif point[1] > rect[3] :
        return False
    return True

def draw_delaunay(img, subdiv, delaunay_color ) :

    triangleList = subdiv.getTriangleList()
    size = img.shape
    r = (0, 0, size[1], size[0])

    for t in triangleList :
        
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])
        
        if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3) :
        
            cv.line(img, pt1, pt2, delaunay_color, 1, cv.LINE_AA, 0)
            cv.line(img, pt2, pt3, delaunay_color, 1, cv.LINE_AA, 0)
            cv.line(img, pt3, pt1, delaunay_color, 1, cv.LINE_AA, 0)

def draw_voronoi(image, subdiv) :

    (facets, centers) = subdiv.getVoronoiFacetList([])

    for i in range(0,len(facets)) :
        ifacet_arr = []
        for f in facets[i] :
            ifacet_arr.append(f)
        ifacet = np.array(ifacet_arr, np.int)
        ifacets = np.array([ifacet])
        cv.polylines(image, ifacets, True, (255, 0, 0), 1, cv.LINE_AA, 0)

def voronoi(points,image):
    img_voronoi = image.copy()
    size = image.shape
    rect = (0, 0, size[1], size[0])
    subdiv = cv.Subdiv2D(rect)
    
    for p in points :
        subdiv.insert(p)
        '''
        img_copy = image.copy()
        # Draw delaunay triangles
        draw_delaunay( img_copy, subdiv, (0, 0, 255) )
        cv.namedWindow('Delaunay Triangulation', cv.WINDOW_NORMAL)
        cv.imshow('Delaunay Triangulation', img_copy)
        cv.waitKey(10)
        '''
    draw_delaunay( image, subdiv, (0, 0, 255) )
    for p in points :
        cv.circle(image, p, 2, (255,0,0), cv.FILLED, cv.LINE_AA, 0 )

    draw_voronoi(img_voronoi,subdiv)
    showImage('Delaunay Triangulation',image)
    showImage('Voronoi Diagram',img_voronoi)
    
points = findCentroids(bin_sauvola,img_centroids)
showImage('Centroids',img_centroids)
voronoi(points, img_voro)    


def k_neighbors_graph(V, k):
    # k: int the number of neighbor to consider for each vector
    # k = len(X)-1 gives the exact MST
    k = min(len(V) - 1, k)
    
    # generate a sparse graph using the k nearest neighbors of each point
    return kneighbors_graph(V, n_neighbors=k, mode='distance')

def minimum_spanning_tree_edges(V, k): #return vector of edges
    
    G = k_neighbors_graph(V, k)
    
    # Compute the minimum spanning tree of this graph
    full_tree = minimum_spanning_tree(G, overwrite=True)
    

    return np.array(full_tree.nonzero()).T 
Graph = k_neighbors_graph(points,5)           
k_kneighbors_edges = np.array(Graph.nonzero()).T
k_kneighbors_distances = Graph.data


ciao=[]
for i in range(k_kneighbors_distances.shape[0]):
    ciao.append(i)
plt.figure(figsize=(20,10))
plt.bar(ciao,k_kneighbors_distances)
plt.show()
#plt.show()
points = np.int32(points)
mst_edges= minimum_spanning_tree_edges(points,5)
'''
distance = []
for edge in edges:
    c1,c2 = edge
    x1,y1 = points[c1]
    x2,y2 = points[c2]
    print(x1,y1,x2,y2)
    dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    distance.append(dist)
    print(dist)
'''  
def angle_between(p1, p2):
    dx = p2[1]-p1[1]
    dy = p2[0]-p1[0]
    #ang1 = np.arctan2(p1[0],p1[1])
    #ang2 = np.arctan2(p2[0],p2[1])
    #return np.rad2deg((ang1 - ang2) % (2 * np.pi))
    arctan = math.atan2(dx,dy)
    return math.degrees(arctan)

plt.figure(figsize=(30,20))
plt.imshow(img, 'gray')    
img_blank = np.zeros(img.shape, np.uint8)
angles = []
for edge in k_kneighbors_edges:
    #i, j = edge
    i,j = edge
    x1,y1 = points[i]
    x2,y2 = points[j]
    angles.append(angle_between(points[i],points[j]))
    cv.line(img_blank, (points[i, 0], points[i, 1]), (points[j, 0], points[j, 1]), (0,0,255), 3, cv.LINE_AA)
    plt.plot([points[i, 0], points[j, 0]], [points[i, 1], points[j, 1]], c='r')
plt.show()   
showImage('K-NN',img_blank) 
oriz =[]
vert = []
for angle in angles:
    abs_angle = abs(angle)
    
    if -45<= angle <= 45 or 135 <= angle or angle <= -135:
        oriz.append(angle)
    else:
        vert.append(angle)

    
#plt.scatter(p[:, 0], p[:, 1])
plt.figure(figsize=(30,20))
plt.imshow(img, 'gray')    
for edge in mst_edges:
    #i, j = edge
    i,j = edge
    x1,y1 = points[i]
    x2,y2 = points[j]
    dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    #print(edge)
    #if dist<25:
    #angles.append(angle_between(points[i],points[j]))
    plt.plot([points[i, 0], points[j, 0]], [points[i, 1], points[j, 1]], c='r')
plt.show()


 
# calculate the Euclidean distance between two vectors
def euclidean_distance(point1, point2):
	distance = (point2[0] - point1[0])**2 + (point2[1] - point1[1])**2
	return math.sqrt(distance)
 
# Locate the most similar neighbors
def get_neighbors(train, test_point, num_neighbors):
    distances = []
    index = 0
    for train_point in train:
        dist = euclidean_distance(test_point, train_point)
        distances.append((train_point, dist, index))
        index +=1
    distances.sort(key=lambda tup: tup[1])
    neighbors = []
    for i in range(num_neighbors):
        neighbors.append((distances[i][0],distances[i][2]))
    return neighbors
 
# Test distance function

neighbors = get_neighbors(points, points[0], 3)
for neighbor in neighbors:
	print(neighbor)


'''
plt.subplot(3,2,4)
plt.imshow(labels, 'nipy_spectral')
plt.title('Connected Components')
plt.axis('off')


plt.subplot(3,2,5)
plt.imshow(img_rlsa_oriz, 'gray')
plt.title('Connected Components')
plt.axis('off')

plt.subplot(3,2,6)
plt.imshow(img_rlsa_vert, 'gray')
plt.title('Connected Components')
plt.axis('off')
'''


    





