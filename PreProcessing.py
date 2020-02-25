import Utils as ut

def binarization(mode,image):
    '''
    Method used to make the binarization, using two different threshold (Otsu and Sauvola) and the inverse binarization
    
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

def iteration(binarized_img, value):
    '''
    Method that perform the RLSA 
    
    Parameters
    ----------
    binarized_img: array of image pixels binarized 
    value: integer of number of pixels to merge doing the RLSA 

    Returns
    -------
    binarized_img: array of image pixels binarized after apply RLSA 
    '''
    rows, cols = binarized_img.shape
    for row in range(rows):
        try:
            start = binarized_img[row].tolist().index(0)
        except ValueError:
            start = 0

        count = start
        for col in range(start, cols):
            if binarized_img[row, col] == 0:
                if (col-count) <= value and (col-count) > 0:
                    binarized_img[row, count:col] = 0               
                count = col  
    return binarized_img


    
def rlsa(binarized_img, horizontal: bool = True, vertical: bool = True, value: int = 0):
    '''
    Method that compute the Run Length Smoothing Algorithm (RLSA) using the method iteration().
    RLSA consists of extracting the block of text or the Region of interest (ROI) from the binary image of the document. 
    It's necessary to pass them binarized image of type ndarray.
    
    Parameters
    ----------
    binarized_img: array of image pixels binarized
    horizontal: boolean value to enable the horizontal RLSA (default = True) (optional) 
    vertical: boolean value to enable the vertical RLSA (default = True) (optional)
    value: integer of number of pixels to merge doing the RLSA
    
    Returns
    -------
    binarized_img: array of image pixels binarized after apply RLSA
    '''
    if horizontal:
            binarized_img = iteration(binarized_img, value)
    if vertical: 
            binarized_img = binarized_img.T
            binarized_img = iteration(binarized_img, value)
            binarized_img = binarized_img.T
    return binarized_img


def showCC(binarized_img, connectivity):
    '''
    Method that shows the connected components and plot them using the library matplotlib
    
    Parameters
    ----------
    binarized_img: array of image pixels binarized
    connectivity: integer (4 or 8) for choosing the number of neaghbors to consider.
    
    '''
    cc, labels = ut.cv.connectedComponents(~binarized_img,connectivity=connectivity)
    
    label_color = ut.np.uint8(179*labels/(cc-1))
    blank_channel = 255*ut.np.ones_like(label_color)#creates a white matrix
    labeled_img = ut.cv.merge([label_color, blank_channel, blank_channel])#creates an image in RGB
    
    labeled_img = ut.cv.cvtColor(labeled_img, ut.cv.COLOR_HSV2BGR)#convert to BGR for display
    
    labeled_img[label_color==0] = 0 #set backgroud label to black
    
    if connectivity==4:
        print('number of CC with connectivity 4:',cc)
        titles = ['Original Image','CC with connettivity 4']
    elif connectivity==8:
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
    
def valueRLSA(binarized_img, vertical: bool = False):
    '''
    Method used to perform the adaptive RLSA calculating the value to use in the method rlsa()
    
    Parameters
    ----------
    binarized_img: array of image pixels binarized
    vertical: boolean value to enable the calculation the horizontal RLSA (default = False) (optional) 
    
    Returns
    -------
    value: integer of final rounded value to perform RLSA
    distances: array of distances between all CCs
    '''
    distances = ut.np.asarray(ut.findDistance(binarized_img, vertical))
    numSum=0
    sumDist=0 
    distances = distances.astype(ut.np.int)
    for k in range(distances.shape[0]):
        sumDist += distances[k]
        numSum += 1  
           
    value=sumDist/numSum
    midW = ut.findMidDistanceContour(binarized_img,vertical)
    
    value -= midW
    return round(value),distances

def houghTransformDeskew(binarized_img, original_img, plot : bool = False):
    '''
    Compute deskew angle using Hough transform, it also plot the histogram of Hough
    transform and build the deskew of the image.
    
    Parameters
    ----------
    binarized_img : array of binarized image pixels.
    original_img: array of original image pixels.
    plot: boolean value used to enable Hough transform histogram plot. (default = False) (optional)

    Returns
    -------
    img_rotated : array of deskewed image pixels, or None if the image is not skewed.
    '''
    
    edges = ut.cv.Canny(binarized_img, 50, 200, 3)#find edges on the image
    img_lines = ut.cv.cvtColor(edges, ut.cv.COLOR_GRAY2BGR) #convert edges image from Gray to BGR

    lines = ut.cv.HoughLinesP(edges, 1, ut.np.pi/180, 100, None, 200, 15) #function used for finding coordinates x0,y0 and x1,y1 for deskew
    tested_angles = ut.np.linspace(0, ut.np.pi , 1080)
    h, theta, d = ut.hough_line(edges,tested_angles)#function used for plot histogram Hough transform
    
    if lines is not None:
        angles = 0.0
        num_lines = len(lines)
        
        for i in range(num_lines):
            l = lines[i][0]
            angle = ut.math.atan2(l[3] - l[1],l[2] - l[0]) * 180.0 / ut.np.pi
            if not (70 < angle < 110 or 250 < angle < 290): 
                ut.cv.line(img_lines, (l[0], l[1]), (l[2], l[3]), (255,0,0), 3, ut.cv.LINE_AA)
                angles += angle
                
        angles /= num_lines #averages between all found angles
        best_angle = angle #find the best angle in the right notation
        
        height, width = original_img.shape[:2]
        center = (width // 2, height // 2)
        
        if plot:
            print('The image is rotated of {:.4f} degrees'.format(best_angle))
            ut.plt.figure(figsize=(20,20))
            ut.plt.imshow(ut.np.log(1 + h), extent=[(ut.np.rad2deg(theta[-1])/2)*-1, ut.np.rad2deg(theta[-1])/2, d[-1], d[0]], cmap ='nipy_spectral', aspect=1.0 / (height/30))
            ut.plt.title('Histogram Hough Transform')
            ut.plt.show()
            
        if best_angle == 0.0:
            return None,None
        #build the deskew
        root_mat = ut.cv.getRotationMatrix2D(center, best_angle, 1)
        img_rotated = ut.cv.warpAffine(original_img, root_mat, (width,height), flags=ut.cv.INTER_CUBIC, borderMode=ut.cv.BORDER_REPLICATE)
        img_rotated_no_fig = ut.cv.warpAffine(binarized_img, root_mat, (width,height), flags=ut.cv.INTER_CUBIC, borderMode=ut.cv.BORDER_REPLICATE) 
        return img_rotated, img_rotated_no_fig
    return None,None

def projection(binarized_img):
    '''
    Method used to count black pixels (img_bin==0) per row (axis=0: col, axis=1:row)
    
    Parameters
    ----------
    binarized_img : array of binarized image pixels.
    
    Returns
    -------
    counts: integer represents number of black pixels
    row_number: integer represent number of rows
    '''
    
    counts = ut.np.sum(binarized_img==0, axis=1) 
    row_number = [i for i in range(binarized_img.shape[0])]
    
    return counts, row_number