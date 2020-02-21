

#import igraph

#from igraph import Graph, EdgeSeq


import Utils as ut
import PreProcessing as pp
import LayoutAnalysis as la


import seaborn as sns


'''
Sistemare voronoi (area) (farlo segmentare )
ocr pytesseract per ogni blocco testuale
commentare e modulare semiDONE (controllare i metodi già fatti)


'''


def main():
    
    img = ut.cv.imread("Immagini prova\\N0024670aau.tif")  #Leggo l'immagine 
    #img = ut.cv.imread("aab.jpg")
    img_orig = img.copy()
    img_word = img.copy()
    img_centroids = img.copy()
    
    #if you want to show every step result on a OpenCV window and write it on a .tif file
    SHOWSTEPS_AND_WRITE_RESULTS = True 
    BINARIZATION_METHOD = 'Sauvola' # or 'otsu','inverse'
    PAGE_SEGMENTATION_METHOD = 'docstrum' #or 'docstrum', 'Voronoi', 'xy tree'
    
    
    if BINARIZATION_METHOD == 'Sauvola':
        binarization = pp.binarization('Sauvola',img)
    elif BINARIZATION_METHOD == 'Otsu':
        binarization = pp.binarization('Otsu',img)
    elif BINARIZATION_METHOD == 'inverse':
        binarization = pp.binarization('inverse',img)
        
    
    '''
    PRE-PROCESSING-------------------------------------------------------------
    '''
    img_no_figures = ut.removeFiguresOrSpots(binarization,'figures')
    img_rotated, img_rotated_no_fig = pp.houghTransformDeskew(img_no_figures,img,plot=True)
    
    if img_rotated is not None: #update image and image without figures
        #showImage('Rotated Image', img_rotated)
        img = img_rotated.copy()
        binarization = pp.binarization('Sauvola',img)
        #ut.showImage('bin rot', bin_rotated)
        img_no_figures = img_rotated_no_fig.copy()
        binarization_no_figures = pp.binarization('Sauvola',img_no_figures)
        
    else:
        print('Image not skewed')
        
    counts_proj, row_number_proj = pp.projection(binarization)
    
    bin_no_spots = ut.removeFiguresOrSpots(binarization,'spots')
    img_no_spots = ut.cv.cvtColor(bin_no_spots, ut.cv.COLOR_GRAY2RGB)
    
    #update image and binarization
    binarization = bin_no_spots.copy()
    img = img_no_spots.copy()
    
    pp.showCC(binarization, 8)  
    
    bin_no_lines = ut.removeFiguresOrSpots(binarization.copy(), 'linesBoth')
    
    img_centroids = img.copy()
    
    points = la.findCentroids(bin_no_lines,img_centroids)
    
    bin_char_box = binarization.copy()
    ut.printContours(bin_char_box,bin_char_box,1)
    
    #Compute adaptive RLSA horizontal and vertical(but vertical is unused)
    value_horiz,dist = pp.valueRLSA(binarization_no_figures)
    
    print('The horizontal adaptive RLSA value is',value_horiz)
    #value_vert, dist = pp.valueRLSA(binarization_no_figures,True)
    #ut.histogram(binarization,dist,True)
    #print(value_vert)
    
    bin_rlsa = pp.rlsa(bin_char_box.copy(), True, False, value_horiz)
    
    img_word = img.copy()
    ut.printContours(bin_rlsa,img_word,1)
    
    '''
    LAYOUT ANALYSIS-----------------------------------------------------
    '''
    
    Graph = la.kNeighborsGraph(points,8)           
    k_kneighbors_edges = ut.np.array(Graph.nonzero()).T
    k_kneighbors_distances = Graph.data
    my_Peak_Values = ut.findPeaks(k_kneighbors_distances,20,True)
    
    if PAGE_SEGMENTATION_METHOD == 'mst':
        _,mst_edges= la.minimumSpanningTreeEdges(points,5,my_Peak_Values)
        img_mst= ut.plotEdges(img,mst_edges,points)
        img_mst_blocks = img.copy()
        mst_coord = ut.printContours(pp.binarization('Sauvola',img_mst),img_mst_blocks,2)
        la.getTextFileFromImage(bin_no_lines,mst_coord,'path','MST_text')
    elif PAGE_SEGMENTATION_METHOD == 'docstrum':
        horizontal_edges, vertical_edges = ut.edgesInformation(k_kneighbors_edges, points, k_kneighbors_distances)
        #Horizontal Docstrum
        bin_docstrum_lines = bin_no_lines.copy()
        la.docstrum(img, bin_docstrum_lines, horizontal_edges, points, max(my_Peak_Values))
        img_docstrum_lines = img_orig.copy()
        ut.printContours(bin_docstrum_lines.copy(),img_docstrum_lines,2)
        
        #Vertical Docstrum
        bin_docstrum_blocks = bin_docstrum_lines.copy()
        la.docstrum(img, bin_docstrum_blocks, vertical_edges, points,  max(my_Peak_Values))
        img_docstrum_blocks = img_orig.copy()
        docstrum_coord = ut.printContours(bin_docstrum_blocks,img_docstrum_blocks,2)
        la.getTextFileFromImage(bin_no_lines,docstrum_coord,'path','docstrum_text')
        
    elif PAGE_SEGMENTATION_METHOD == 'Voronoi':
        img_delaunay = img.copy()
        bin_voro_edges, img_voro_full,img_voro_segmentation = la.voronoi(points, img_delaunay, my_Peak_Values)
        voro_coord = ut.printContours(bin_voro_edges,bin_voro_edges.copy(),1)
        la.getTextFileFromImage(bin_no_lines,voro_coord,'path','voronoi_text')
        
        
    elif PAGE_SEGMENTATION_METHOD == 'xy tree':
        img_name = 'XY_Tree\\Prova0.tif'
        img = ut.cv.imread(img_name)
        path = img_name[:-10]
        img_bin = pp.binarization('otsu', img)
        info=la.cutImage(img_bin,50,40)
        infoV=la.cutImage(img_bin,50,20,True)
        XY_Tree = []
        la.cutMatrix(img_name,path, img_bin, info, infoV, XY_Tree)
        namesfile='*_finalCut*.tif'
        outputfile='SaveText.txt'
        la.getTextFile(path,namesfile,outputfile)
        
        
    if SHOWSTEPS_AND_WRITE_RESULTS:
        
        '''
        ut.showImage('Original Image',img)
        name = BINARIZATION_METHOD + ' Binarization'
        ut.showImage(name,binarization)
        ut.showImage('Rotated Image',img_rotated)
        ut.showProjection(binarization,counts_proj,row_number_proj)
        ut.showImage('Image Without Spots',img_no_spots)
        ut.showImage('Image Without Lines',bin_no_lines)
        ut.showImage('Centroids of Connected Components',img_centroids)
        ut.showImage('Bounding boxes of Connected Components',bin_char_box)
        ut.histogram(binarization,dist)
        ut.showImage('Image after apply RLSA',bin_rlsa)
        '''
        if PAGE_SEGMENTATION_METHOD == 'mst':
            ut.showImage('Minimun Spanning Tree Edges',img_mst)
            ut.showImage('Minimun Spanning Tree Edges Blocks',img_mst_blocks)
        elif PAGE_SEGMENTATION_METHOD == 'docstrum':
            ut.showImage('K-Nearest Neighbors Lines',bin_docstrum_lines) 
            ut.showImage('Docstrum lines',img_docstrum_lines)
            ut.showImage('K-Nearest Neighbors blocks',bin_docstrum_blocks) 
            ut.showImage('Docstrum blocks',img_docstrum_blocks)
        elif PAGE_SEGMENTATION_METHOD == 'Voronoi':
            ut.showImage('Delaunay Triangulation',img_delaunay)
            ut.showImage('Voronoi Edges',bin_voro_edges)
            ut.showImage('Voronoi Diagram',img_voro_full)
            ut.showImage('Voronoi Diagram Segmented',img_voro_segmentation)
        
    return
    
    
    img_name = 'XY_Tree\\Prova0.tif'
    img = ut.cv.imread(img_name)
    path = img_name[:-10]
    img_bin = pp.binarization('otsu', img)
    info=la.cutImage(img_bin,50,40)
    infoV=la.cutImage(img_bin,50,20,True)
    XY_Tree = []
    #pippo, typeNode, label=cutMatrix(img_name, img_bin, info, infoV, pippo)
    la.cutMatrix(img_name,path, img_bin, info, infoV, XY_Tree)
    namesfile='*_finalCut*.tif'
    outputfile='SaveText.txt'
    la.getTextFile(path,namesfile,outputfile)
    
    

if __name__ == "__main__":
    main()


    





