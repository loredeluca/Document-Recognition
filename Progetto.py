import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from skimage.filters import (threshold_otsu, threshold_sauvola)
from skimage import measure
from pythonRLSA import rlsa
import sklearn.preprocessing 

#img = cv.imread("N0024670aao.tif")  #Leggo l'immagine in scala di grigi

img = cv.imread("aac.jpg")
img2 = img.copy()  
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

cv.namedWindow('Immagine Originale', cv.WINDOW_NORMAL)
cv.imshow('Immagine Originale', img)
cv.imwrite('Immagine Originale.png', img)
cv.waitKey(0)
cv.destroyWindow('Immagine Originale')

cv.namedWindow('Binarizzazione con Otsu', cv.WINDOW_NORMAL)
cv.imshow('Binarizzazione con Otsu', bin_otsu.astype(float))
cv.imwrite('Binarizzazione con Otsu.png', bin_otsu.astype(float))
cv.waitKey(0)
cv.destroyWindow('Binarizzazione con Otsu')

#bin_sauvola = cv.fastNlMeansDenoising(bin_sauvola,None,1,7,21)
bin_sauvola = cv.medianBlur(bin_sauvola, 3)

cv.namedWindow('Binarizzazione con Sauvola', cv.WINDOW_NORMAL)
cv.imshow('Binarizzazione con Sauvola', bin_sauvola.astype(float))
cv.imwrite('Binarizzazione con Sauvola.png', bin_sauvola.astype(float))
cv.waitKey(0)
cv.destroyWindow('Binarizzazione con Sauvola')

labels = measure.label(bin_sauvola, background = 1, connectivity=2)

plt.figure(figsize=(30,20))
plt.imshow(labels, 'nipy_spectral')
plt.title('Componenti Connesse')
plt.axis('off')

contours,_  = cv.findContours(np.uint8(np.logical_not(bin_sauvola)),cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE) 

for contour in contours:
    #disegna un rettangolo verde intorno ai caratteri
    [x,y,w,h] = cv.boundingRect(contour)
    cv.rectangle(img, (x,y), (x+w,y+h), (0, 255, 0), 1)

cv.namedWindow('Bordi Caratteri', cv.WINDOW_NORMAL)
cv.imshow('Bordi Caratteri', img)
cv.imwrite('Bordi Caratteri.png', img)
cv.waitKey(0)
cv.destroyAllWindows()

img_rlsa_oriz = rlsa.rlsa(bin_sauvola.copy(), True, False, 15)
img_rlsa_full = rlsa.rlsa(img_rlsa_oriz.copy(), False, True, 10)

contours,_  = cv.findContours(np.uint8(np.logical_not(img_rlsa_full)),cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE) 

for contour in contours:
    #disegna un rettangolo verde intorno alle parole
    [x,y,w,h] = cv.boundingRect(contour)
    cv.rectangle(img2, (x,y), (x+w,y+h), (0, 255, 0), 1)

cv.namedWindow('Bordi Parole', cv.WINDOW_NORMAL)
cv.imshow('Bordi Parole', img2)
cv.imwrite('Bordi Parole.png', img2)
cv.waitKey(0)
cv.destroyAllWindows()

print ("Number of components:", np.max(labels))
print ("Number of components:", np.max(bin_sauvola))

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







