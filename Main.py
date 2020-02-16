

#import igraph

#from igraph import Graph, EdgeSeq


import Utils as ut
import PreProcessing as pp
import LayoutAnalysis as la


import seaborn as sns
from collections import defaultdict  
from scipy.signal import find_peaks
import heapq

'''
Sistemare voronoi (area) (farlo segmentare ehehehehehe)
ocr pytesseract per ogni blocco testuale
commentare e modulare semiDONE (controllare i metodi già fatti)


'''

def findPeaks(k_kneighbors_distances):
    d = defaultdict(int)
    
    for k in k_kneighbors_distances:
        k = round(k)
        d[k] += 1
        
    result = sorted(d.items(), key = lambda x:x[1], reverse=True)
    #result = []
    
    #result = [(23.0, 995), (24.0, 943), (25.0, 719), (22.0, 604), (21.0, 520), (20.0, 438), (19.0, 397), (8.0, 358), (9.0, 356), (17.0, 328), (26.0, 328), (10.0, 326), (18.0, 280), (15.0, 268), (12.0, 266), (16.0, 250), (11.0, 232), (13.0, 232), (14.0, 196), (7.0, 188), (27.0, 175), (28.0, 100), (29.0, 98), (6.0, 82), (30.0, 72), (31.0, 50), (34.0, 42), (33.0, 40), (32.0, 39), (35.0, 39), (36.0, 29), (49.0, 24), (44.0, 22), (38.0, 20), (37.0, 19), (40.0, 19), (51.0, 16), (41.0, 14), (5.0, 14), (55.0, 14), (45.0, 13), (47.0, 13), (53.0, 13), (58.0, 12), (42.0, 12), (39.0, 12), (48.0, 12), (52.0, 11), (50.0, 10), (43.0, 8), (54.0, 7), (56.0, 7), (57.0, 5), (70.0, 3), (46.0, 3), (77.0, 3), (76.0, 2), (72.0, 2), (61.0, 2), (4.0, 2)]
    print(result)
    
    values = []
    occurrences = []
    for i in range(len(result)):
        occurrences.append(result[i][1])
        values.append(result[i][0])

    x = ut.np.array(values)
    y = ut.np.array(occurrences)
    
    # orina i dati dal valore più grande al valore piu piccolo(sia rispetto a x sia rispetto a y)
    # x = [8, 3, 5] diventa x = [3, 5, 8]
    sortId = ut.np.argsort(x)
    x = x[sortId]
    y = y[sortId]
    print('x ',x)
    print('y ',y)
    #Trovo i punti di ottimo locale
    peaks, _ = find_peaks(y, distance = 30)

    print('peaks', peaks)
    
    peaksOccurrenceList=[]
    peaksOccurrences=[]
    for i in range(len(peaks)):
        peaksOccurrences.append(y[peaks[i]])
        peaksOccurrenceList.append((x[peaks[i]],y[peaks[i]]))
    peaksOccurrenceList = sorted(peaksOccurrenceList, key = lambda x:x[1], reverse=True)
    print('peackOccurrences',peaksOccurrences)

    #Trova i DUE picchi piu alti 
    bestPeaks=heapq.nlargest(2, peaksOccurrences)
    print(bestPeaks)


    my_Peak_Value=[]
    for k in range(len(peaksOccurrenceList)):
        if bestPeaks[0]==peaksOccurrenceList[k][1] or bestPeaks[1]==peaksOccurrenceList[k][1]:
            my_Peak_Value.append(int(peaksOccurrenceList[k][0]))  
        
    print(my_Peak_Value)
    
    # PLOT
    ut.plt.plot(x, y)
    ut.plt.plot(my_Peak_Value, bestPeaks, "x")
    ut.plt.xlim(0, 50)
    ut.plt.show()
    
    return my_Peak_Value


def main():
    
    img = ut.cv.imread("Immagini prova\\N0024670aan.tif")  #Leggo l'immagine 
    #img = ut.cv.imread("journal1.jpg")
    img_orig = img.copy()
    img_word = img.copy()
    img_centroids = img.copy()
    SHOWSTEPS = True
    
    
    bin_otsu = pp.binarization('otsu',img)
    bin_sauvola = pp.binarization('sauvola',img)
    bin_inv = pp.binarization('inverse',img)
    
    if SHOWSTEPS:
        
        '''
        ut.showImage('Original Image',img)
        ut.showImage('Inverse Binarization',bin_inv)
        ut.showImage('Otsu Binarization',bin_otsu)
        ut.showImage('Sauvola Binarization', bin_sauvola)
        '''
    #bin_sauvola = bin_otsu        
    #bin_sauvola = cv.medianBlur(bin_sauvola, 3) #apply median blur to remove black spots on images
    #showImage('Sauvola Binarization BLUR', bin_sauvola)
    
    #rotated = ut.rotate(img_orig,bin_sauvola)
    #bin_rot = pp.binarization('otsu',rotated)
    
    
    bin_no_spots = ut.removeFiguresOrSpots(bin_sauvola,'spots')
    img_no_spots = ut.cv.cvtColor(bin_no_spots, ut.cv.COLOR_GRAY2RGB)
    #bin_no_spots = pp.binarization('sauvola',img_no_spots)
    
    pp.showCC(bin_no_spots, 8)
    
    
    img = img_no_spots.copy()
    bin_sauvola = bin_no_spots.copy()
    
    
    
    img_bin_char = bin_sauvola.copy()
    points = la.findCentroids(img_bin_char,img_centroids)
    #ut.showImage('Centroids',img_centroids)
    
    ut.plt.figure(figsize=(20,20))
    ut.plt.imshow(img_centroids, 'gray')
    ut.plt.show()
    
    img_bin_char = ut.removeFiguresOrSpots(img_bin_char, 'linesBoth')
    #ut.showImage('Characters Contour no lines', img_bin_char)
    
    ut.printContours(img_bin_char,img_bin_char,1)    
    ut.showImage('Characters Contour', img_bin_char)
    
    
    '''
    points = la.findCentroids(img_bin_char,img_centroids)
    ut.showImage('Centroids',img_centroids)
    '''
    
    img_no_figures2 = ut.removeFiguresOrSpots(bin_sauvola,'figures')
    #ut.showImage('nofig2',img_no_figures2)
    
    #RISOLTO PROBLEMA RLSA ADATTIVO
    valueH,dist = pp.valueRLSA(pp.binarization('sauvola',img_no_figures2))
    ut.histogram(bin_sauvola,dist)
    print(valueH)
    valueV, dist = pp.valueRLSA(pp.binarization('sauvola',img_no_figures2),True)
    ut.histogram(bin_sauvola,dist,True)
    print(valueV)
    
    img_rlsa_H = pp.rlsa(img_bin_char.copy(), True, False, valueH)
    
    img_rlsa_full = pp.rlsa(img_rlsa_H.copy(), False, True, valueV)

    #ut.showImage('RLSA',img_rlsa_full)
    ut.printContours(img_rlsa_full,img_word,1)
    ut.showImage('Words Contour', img_word)
    
    #img_no_figures = ut.removeFiguresOrSpots(img_rlsa_full,'figures')
    #ut.showImage('Image Without Figures', img_no_figures2)
 
    img_rotated, img_rotated_no_fig = pp.houghTransformDeskew(img_no_figures2,img)

    if img_rotated is not None:
        #showImage('Rotated Image', img_rotated)
        bin_rotated = pp.binarization('sauvola',img_rotated)
        #ut.showImage('bin rot', bin_rotated)
        bin_rotated_no_fig = pp.binarization('sauvola',img_rotated_no_fig)
        img = img_rotated.copy()
    else:
        print('Image not skewed')

    img_voro = img_no_spots.copy() 
    img_k = img_no_spots.copy()
    img_docstrum_lines = img_orig.copy()
    img_docstrum_box = img_orig.copy()
    
    #projection(bin_rotated,rotated)
    '''
    ut.showImage('noff',img_rotated_no_fig)
    points = la.findCentroids(bin_rotated_no_fig,img_centroids)
    ut.showImage('Centroids',img_centroids)
    '''
    
    '''
    _,voro,voro_points, voro_blank_inv = la.voronoi(points, img_voro)
    
    contours,_ = ut.cv.findContours(voro_blank_inv, ut.cv.RETR_TREE, ut.cv.CHAIN_APPROX_SIMPLE)
    
    
    blank_img = ut.np.ones([img.shape[0],img.shape[1]],dtype=ut.np.uint8)*255
    voro_blank_inv = ut.cv.drawContours(blank_img, contours, -1 , (0,255,0),1)
    ut.showImage('pippp',blank_img)
    areas = []
    for c in contours:
        area = ut.cv.contourArea(c)
        areas.append(area)
        
    
    
    #print(areas)
    ut.plt.imshow(voro,'gray')
    ut.plt.show()
    
    '''
    
    
    '''
    voro_graph = la.kNeighborsGraph(voro_points,10)
    voro_k_kneighbors_edges = ut.np.array(voro_graph.nonzero()).T
    #voro_k_kneighbors_distances = voro_graph.data
    voro_points = ut.np.int32(voro_points)
    ut.plotEdges(img,voro_k_kneighbors_edges,voro_points)
    '''
    Graph = la.kNeighborsGraph(points,3)           
    k_kneighbors_edges = ut.np.array(Graph.nonzero()).T
    k_kneighbors_distances = Graph.data
    
    media_dist = ut.stat.mean(k_kneighbors_distances)
    print('media',media_dist)
    print('media +',media_dist+(media_dist/2))
    print('media -',media_dist-(media_dist/2))
    
    
    
    
    #print('PEAKS', peaks)
    
    my_Peak_Value = findPeaks(k_kneighbors_distances)
    
    #temporaneo
    #points = ut.np.int32(points)
    #ut.plotEdges(img,k_kneighbors_edges,points)
        
    
    
    #kwargs = dict(hist_kws={'alpha':.6}, kde_kws={'linewidth':2})
    ut.plt.figure(figsize=(20,10))
    #ut.plt.hist(k_kneighbors_distances, bins=1000)
    sns.distplot(k_kneighbors_distances, color="dodgerblue", bins=300)
    ut.plt.xlim(0,100)
    ut.plt.gca().set(title='Frequency Histogram', ylabel='Frequency')
    ut.plt.show()
    
    
    
    row_number = [i for i in range(k_kneighbors_distances.shape[0])]
    ut.plt.figure(figsize=(20,10))
    ut.plt.bar(row_number,k_kneighbors_distances)
    ut.plt.show()
    
    #return

    _,mst_edges= la.minimumSpanningTreeEdges(points,5)
    
    points = ut.np.int32(points)

    print(k_kneighbors_edges)
    ut.plotEdges(img,k_kneighbors_edges,points)
    ut.plotEdges(img,mst_edges,points)
    
    horizontal_edges, vertical_edges = ut.edgesInformation(k_kneighbors_edges, points, k_kneighbors_distances)
    
    oriz=[]
    vert=[]
    '''
    for e in horizontal_edges:
        oriz.append(e[2])
    for e in vertical_edges:
        vert.append(e[2])
    print('orizz',oriz)
    
    ut.plotEdges(img,oriz,points)
    print('vert')
    ut.plotEdges(img,vert,points)
    
    distanzeOV = []
    for e in horizontal_edges:
        oriz.append(e[1])
    for e in vertical_edges:
        vert.append(e[1])
        
    print(oriz)
    findPeaks(oriz)
    print(vert)
    findPeaks(vert)
    '''
    #DOCSTRUM ORIZZONTALE
    la.docstrum(img, img_k, horizontal_edges, points, my_Peak_Value[1])
    
    img_k_orig = img_k.copy()

    ut.showImage('K-NN Lines',img_k) 
    bin_img_k = pp.binarization('sauvola',img_k)
    rlsa_docstrum = pp.rlsa(bin_img_k.copy(), True, False, valueH)
    ut.showImage('K-NN Lines rlsa',rlsa_docstrum) 
    ut.printContours(rlsa_docstrum,img_docstrum_lines,2)
    ut.showImage('Docstrum lines',img_docstrum_lines)
    
    #DOCSTRUM VERTICALE
    la.docstrum(img, img_k, vertical_edges, points,  my_Peak_Value[1])
    
    ut.showImage('K-NN blocks',img_k) 
    bin_img_k = pp.binarization('sauvola',img_k)
    rlsa_docstrum = pp.rlsa(bin_img_k.copy(), False, True, valueV)
    ut.showImage('K-NN blocks rlsa',rlsa_docstrum) 
    ut.printContours(rlsa_docstrum,img_docstrum_box,2)
    ut.showImage('Docstrum blocks',img_docstrum_box)
    
def main1():
    
    img_name = 'XY_Tree\\Prova0.tif'
    img = ut.cv.imread(img_name)
    
    img_bin = pp.binarization('otsu', img)
    #path = 'C:\\Users\\manus\\Desktop\\Progetto DDM\\XY_Tree\\'
    path = img_name[:-10]
    info=la.cutImage(img_bin,50,40)
    infoV=la.cutImage(img_bin,50,20,True)
    XY_Tree = []
    #pippo, typeNode, label=cutMatrix(img_name, img_bin, info, infoV, pippo)
    la.cutMatrix(img_name,path, img_bin, info, infoV, XY_Tree)
    '''
    XY_Tree.vs["type"] = typeNode
    XY_Tree.vs["label"] = label
    color_dict = {"root": "white", "leaf": "red", "node": "yellow"}
    XY_Tree.vs["color"] = [color_dict[type] for type in XY_Tree.vs["type"]]

    print(XY_Tree)
    #layout = pippo.layout("tree")
    layout = XY_Tree.layout_reingold_tilford(mode="in", root=0)
    igraph.plot(XY_Tree, layout = layout, bbox = (200, 200), margin = 20)
    '''

def main2():
    '''
    path = 'Immagini prova\\'
    filelist = []
    for infile in glob.glob (os.path.join (path, 'N0024670*.tif')):
        filelist.append (infile)
    for file in filelist:
        '''
    rot = ut.rotate(ut.cv.imread('Immagini prova\\N0024670aas.tif'),pp.binarization('otsu',ut.cv.imread('Immagini prova\\N0024670aas.tif')))
    ut.showImage('pippo',rot)

if __name__ == "__main__":
    main()


    





