#from skimage.filters import (threshold_otsu, threshold_sauvola)
#from sklearn.preprocessing import binarize

import Utils as ut

def binarization(mode,image):
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
        image = ut.cv.cvtColor(image, ut.cv.COLOR_BGR2GRAY)
    if mode == 'Otsu':
        return ut.binarize(image, ut.threshold_otsu(image))*255
    elif mode == 'Sauvola':
        return ut.binarize(image, ut.threshold_sauvola(image))*255
    elif mode == 'inverse':
        _,thresh = ut.cv.threshold(image,127,255,ut.cv.THRESH_BINARY_INV)
        return thresh

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
    
    cc, labels = ut.cv.connectedComponents(~binarized_img,connectivity=conn)
    
    label_color = ut.np.uint8(179*labels/(cc-1))
    blank_channel = 255*ut.np.ones_like(label_color)#creates a white matrix
    labeled_img = ut.cv.merge([label_color, blank_channel, blank_channel])#creates an image in RGB
    
    labeled_img = ut.cv.cvtColor(labeled_img, ut.cv.COLOR_HSV2BGR)#convert to BGR for display
    
    labeled_img[label_color==0] = 0 #set backgroud label to black
    
    if conn==4:
        print('number of CC with connectivity 4:',cc)
        titles = ['Original Image','CC with connettivity 4']
    elif conn==8:
        print('number of CC with connectivity 8:',cc)
        titles = ['Original Image','CC with connettivity 8']
    images = [binarized_img, labeled_img]
    for i in range(2):
        ut.plt.figure(figsize=(20,20))
        ut.plt.subplot(1,2,i+1)
        ut.plt.imshow((images[i]),'gray')
        ut.plt.title(titles[i])
        ut.plt.axis('off')
    ut.plt.show()
    
def valueRLSA(binarized_img, vert: bool = False):
    
    distances = ut.np.asarray(ut.findDistance(binarized_img, vert))
    numSum=0
    sumDist=0 
    distances = distances.astype(ut.np.int)
    for k in range(distances.shape[0]):
        sumDist += distances[k]
        numSum += 1  
           
    value=sumDist/numSum
    midW = ut.findMidDistanceContour(binarized_img,vert)
    
    value -= midW
    return round(value),distances

def houghTransformDeskew(binarized_img,original_img, plot : bool = False):
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
    
    edges = ut.cv.Canny(binarized_img, 50, 200, 3)#find edges on the image
    img_lines = ut.cv.cvtColor(edges, ut.cv.COLOR_GRAY2BGR) #convert edges image from Gray to BGR

    lines = ut.cv.HoughLinesP(edges, 1, ut.np.pi/180, 80, None, 100, 10) #function used for finding coordinates x0,y0 and x1,y1 for deskew
    tested_angles = ut.np.linspace(0, ut.np.pi , 1080)
    h, theta, d = ut.hough_line(edges,tested_angles)#function used for plot histogram Hough transform
    
    if lines is not None:
        angle = 0.0
        num_lines = len(lines)
        
        for i in range(num_lines):
            #write blue lines on image according to Hough lines
            l = lines[i][0]
            ut.cv.line(img_lines, (l[0], l[1]), (l[2], l[3]), (255,0,0), 3, ut.cv.LINE_AA)
            #angle += ut.math.atan2(l[3]*1.0 - l[1]*1.0,l[2]*1.0 - l[0]*1.0)
            angle += ut.math.atan2(l[3] - l[1],l[2] - l[0])
           
        angle /= num_lines #averages between all found angles
        best_angle = angle* 180.0 / ut.np.pi #find the best angle in the right notation
        
         
        #ut.showImage('Detected Lines with Probabilistic Line Transform', img_lines)#non so se puo essere utile ai fini del progetto stampare le linee 
        
        height, width = original_img.shape[:2]
        center = (width // 2, height // 2)
        
        if plot:
            print('The image is rotated of {:.2f} degrees'.format(best_angle))
            #show histogram of Hough transform
            ut.plt.figure(figsize=(20,20))
            ut.plt.imshow(ut.np.log(1 + h), extent=[(ut.np.rad2deg(theta[-1])/2)*-1, ut.np.rad2deg(theta[-1])/2, d[-1], d[0]], cmap ='nipy_spectral', aspect=1.0 / (height/30))
            ut.plt.title('Histogram Hough Transform')
            ut.plt.show()
        #build the deskew
        root_mat = ut.cv.getRotationMatrix2D(center, best_angle, 1)
        img_rotated = ut.cv.warpAffine(original_img, root_mat, (width,height), flags=ut.cv.INTER_CUBIC, borderMode=ut.cv.BORDER_REPLICATE)
        img_rotated_no_fig = ut.cv.warpAffine(binarized_img, root_mat, (width,height), flags=ut.cv.INTER_CUBIC, borderMode=ut.cv.BORDER_REPLICATE) 
        return img_rotated, img_rotated_no_fig
    return None

def projection(img_bin):
    
    #Counting black pixels (img_bin==0) per row (axis=0: col, axis=1:row)
    counts = ut.np.sum(img_bin==0, axis=1) 
    row_number = [i for i in range(img_bin.shape[0])]
    #counts = smooth(counts,20) #ammorbidisce il grafico dei pixel
    return counts, row_number