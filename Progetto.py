import cv2 as cv
import numpy as np
import math
from matplotlib import pyplot as plt
from skimage.filters import (threshold_otsu, threshold_sauvola)
from skimage.transform import hough_line
from skimage import measure
from pythonRLSA import rlsa
import sklearn.preprocessing 


#img = cv.imread("N0024670aao.tif")  #Leggo l'immagine 

img = cv.imread("skew.tif")
img_char,img_word = img.copy(),img.copy()  
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

bin_otsu = sklearn.preprocessing.binarize(img_gray, threshold_otsu(img_gray))
bin_sauvola = sklearn.preprocessing.binarize(img_gray, threshold_sauvola(img_gray))
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
    cv.namedWindow(text_name, cv.WINDOW_NORMAL)
    cv.imshow(text_name, file_name)
    cv.imwrite(text_name+'.png', file_name)
    cv.waitKey(0)
    cv.destroyWindow(text_name)
    
showImage('Original Image',img)
showImage('Otsu Binarization',bin_otsu*255)

bin_sauvola = cv.medianBlur(bin_sauvola, 5) #apply median blur to remove black spots on image

showImage('Sauvola Binarization', bin_sauvola*255)

labels = measure.label(bin_sauvola, background = 1, connectivity=2)

plt.figure(figsize=(20,20))
plt.imshow(labels, 'nipy_spectral')
plt.title('Connected Components')
plt.axis('off')
plt.show()

def printContours(binarization,output_img):
    #draw a green rectangle around to characters/words
    contours,_  = cv.findContours(np.uint8(np.logical_not(binarization)),cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE) 
    
    for contour in contours:
    
        #disegna un rettangolo verde intorno ai caratteri
        [x,y,w,h] = cv.boundingRect(contour)
        cv.rectangle(output_img, (x,y), (x+w,y+h), (0, 255, 0), 1)
        
        
printContours(bin_sauvola,img_char)    
showImage('Characters Contour', img_char)    

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
            '''if image[i][j]==255 and image[i][0]==255:
                if i!=i2:
                    flag=1
                    i2=i
                else:
                    flag=flag+1
                    i2=i'''
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

pix_dist_horiz=pixelDistance(bin_otsu)
pix_dist_vert=pixelDistance(bin_otsu,True)
value_horiz=valueRLSA(pix_dist_horiz)
value_vert=valueRLSA(pix_dist_vert)
histogram(bin_sauvola,pix_dist_horiz)
histogram(bin_sauvola,pix_dist_vert)

#DA VERIFICARE RLSA ADATTIVO, HO FATTO VARI TEST MA LA DIVISIONE DELLE PAROLE NON FUNZIONA BENE

img_rlsa_oriz = rlsa.rlsa(bin_sauvola.copy(), True, False, value_horiz)  
img_rlsa_full = rlsa.rlsa(img_rlsa_oriz.copy(), False, True, value_vert)


showImage('RLSA',img_rlsa_full*255)
printContours(img_rlsa_full,img_word)
showImage('Words Contour', img_word)   

def houghTransformDeskew(binarized_img,original_img):
  
    edges = cv.Canny(binarized_img*255, 50, 200, 3)#find edges on the image
    img_lines = cv.cvtColor(edges, cv.COLOR_GRAY2BGR) #convert edges image from Gray to BGR

    lines = cv.HoughLinesP(edges, 1, np.pi/180, 80,None,100,10) #function used for finding coordinates x0,y0 and x1,y1 for deskew
    tested_angles = np.linspace(-np.pi/2, np.pi / 2, 360)
    h, theta, d = hough_line(edges,tested_angles)#function used for plot histogram Hough transform

    #show histogram
    
    
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
        
        plt.figure(figsize=(20,20))
        plt.imshow(np.log(1 + h), extent=[np.rad2deg(theta[-1]), np.rad2deg(theta[0]), d[-1], d[0]],cmap ='nipy_spectral',aspect=1.0 / (height/16))
        plt.title('Histogram Hough Transform')
        
        root_mat = cv.getRotationMatrix2D(center, best_angle, 1)
        rotated = cv.warpAffine(original_img, root_mat, (width,height), flags=cv.INTER_CUBIC,borderMode=cv.BORDER_REPLICATE) 
        return rotated
    return None
    
rotated = houghTransformDeskew(img_rlsa_full,img)
if rotated is not None:
    showImage('Rotated Image', rotated)
else:
    print('Image not skewed')

#showImage('Detected Lines with Probabilistic Line Transform', img_lines)



'''
plt.subplot(3,2,4)
plt.imshow(labels, 'nipy_spectral')s
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


    





