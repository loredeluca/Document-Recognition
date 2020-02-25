import Utils as ut
import PreProcessing as pp
import LayoutAnalysis as la


def main():
    
    input_path = 'input images\\'
    output_path = 'output images\\'
        
    #if you want to show every step result on a OpenCV window and write it on a .tif file
    SHOWSTEPS_AND_WRITE_RESULTS = True 
    BINARIZATION_METHOD = 'inverse' # or 'otsu','inverse'
    PAGE_SEGMENTATION_METHOD = 'Voronoi' #or 'docstrum', 'Voronoi', 'xy tree'
    
    
    #read dataset and put it in an array
    file_list = []
    for input_file in (ut.os.listdir (input_path)):
        file_list.append (input_file) 
        
    for file_name in file_list:
        img_name = file_name
        img = ut.cv.imread(input_path+img_name)  #read image 
        img_orig = img.copy() #keep a reference to the original image
    
    
        '''
        PRE-PROCESSING-------------------------------------------------------------
        '''
    
        if BINARIZATION_METHOD == 'Sauvola':
            binarization = pp.binarization('Sauvola',img)
        elif BINARIZATION_METHOD == 'Otsu':
            binarization = pp.binarization('Otsu',img)
        elif BINARIZATION_METHOD == 'inverse':
            binarization = pp.binarization('inverse',img)
    
        
        
        img_no_figures = ut.removeFiguresOrSpots(binarization,'figures')
        binarization_no_figures = pp.binarization('Sauvola', img_no_figures)
        img_rotated, img_rotated_no_fig = pp.houghTransformDeskew(img_no_figures, img, plot=True)
        
        if img_rotated is not None: #update image and image without figures
            
            img = img_rotated.copy()
            binarization = pp.binarization('Sauvola',img)
            img_no_figures = img_rotated_no_fig.copy()
            binarization_no_figures = pp.binarization('Sauvola',img_no_figures)
        
        else:
            img_rotated = img.copy()
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
        value_horiz,distances = pp.valueRLSA(binarization_no_figures)
        print('The horizontal adaptive RLSA value is',value_horiz)
        #value vertical-----------------------
        #value_vert, dist = pp.valueRLSA(binarization_no_figures,True)
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
            la.getTextFileFromImage(bin_no_lines,output_path + img_name[:-4] + ' MST_text.txt',mst_coord)
        if PAGE_SEGMENTATION_METHOD == 'docstrum':
            horizontal_edges, vertical_edges = ut.edgesInformation(k_kneighbors_edges, points, k_kneighbors_distances)
            #Horizontal Docstrum
            bin_docstrum_lines = bin_no_lines.copy()
            la.docstrum(bin_docstrum_lines, horizontal_edges, points, max(my_Peak_Values))
            img_docstrum_lines = img_rotated.copy()
            ut.printContours(bin_docstrum_lines.copy(),img_docstrum_lines,2)
        
            #Vertical Docstrum
            bin_docstrum_blocks = bin_docstrum_lines.copy()
            la.docstrum(bin_docstrum_blocks, vertical_edges, points,  max(my_Peak_Values))
            img_docstrum_blocks = img_rotated.copy()
            docstrum_coord = ut.printContours(bin_docstrum_blocks,img_docstrum_blocks,2)
            la.getTextFileFromImage(bin_no_lines, output_path + img_name[:-4] + ' docstrum_text.txt',docstrum_coord)
        
        if PAGE_SEGMENTATION_METHOD == 'Voronoi':
            img_delaunay = img.copy()
            bin_voro_edges, img_voro_full,img_voro_segmentation = la.voronoi(points, img_delaunay, my_Peak_Values)
            voro_coord = ut.printContours(bin_voro_edges,bin_voro_edges.copy(),1)
            la.getTextFileFromImage(bin_no_lines,output_path + img_name[:-4] +  ' voronoi_text.txt',voro_coord)
        
        
        if PAGE_SEGMENTATION_METHOD == 'xy tree':
        
            path = 'XY_Tree\\'
            img_bin = pp.binarization('Otsu', img_rotated)
            info=la.cutImage(img_bin,50,40)
            infoV=la.cutImage(img_bin,50,20,True)
            XY_Tree = []
            immagini = la.cutMatrix(img_name,path, img_bin, info, infoV, XY_Tree)
            la.getTextFileFromImage(immagini, output_path + img_name[:-4] + 'docstrum text.txt',[])
        
        
        
        if SHOWSTEPS_AND_WRITE_RESULTS:
        
            
            ut.showImage('Original Image',img,img_name[:-4],output_path,write=True)
            name = BINARIZATION_METHOD + ' Binarization'
            ut.showImage(name,binarization,img_name[:-4],output_path,write=True)
            ut.showImage('Rotated Image',img_rotated,img_name[:-4],output_path,write=True)
            ut.showProjection(binarization,counts_proj,row_number_proj)
            ut.showImage('Image Without Figures',img_no_figures,img_name[:-4],output_path,write=True)#da eliminare
            ut.showImage('Image Without Spots',img_no_spots,img_name[:-4],output_path,write=True)
            ut.showImage('Image Without Lines',bin_no_lines,img_name[:-4],output_path,write=True)
            ut.showImage('Centroids of Connected Components',img_centroids,img_name[:-4],output_path,write=True)
            ut.showImage('Bounding boxes of Connected Components',bin_char_box,img_name[:-4],output_path,write=True)
            ut.histogram(binarization,distances)
            ut.showImage('Image after apply RLSA',bin_rlsa,img_name[:-4],output_path,write=True)
            ut.showImage('Bounding boxes of Words',img_word,img_name[:-4],output_path,write=True)
            
            if PAGE_SEGMENTATION_METHOD == 'mst':
                ut.showImage('Minimun Spanning Tree Edges',img_mst,img_name[:-4],output_path,write=True)
                ut.showImage('Minimun Spanning Tree Edges Blocks',img_mst_blocks,img_name[:-4],output_path,write=True)
            if PAGE_SEGMENTATION_METHOD == 'docstrum':
                ut.showImage('K-Nearest Neighbors Lines',bin_docstrum_lines,img_name[:-4],output_path,write=True) 
                ut.showImage('Docstrum lines',img_docstrum_lines,write=True)
                ut.showImage('K-Nearest Neighbors blocks',bin_docstrum_blocks,img_name[:-4],output_path,write=True) 
                ut.showImage('Docstrum blocks',img_docstrum_blocks,img_name[:-4],output_path,write=True)
            if PAGE_SEGMENTATION_METHOD == 'Voronoi':
                ut.showImage('Delaunay Triangulation',img_delaunay,img_name[:-4],output_path,write=True)
                ut.showImage('Voronoi Edges',bin_voro_edges,img_name[:-4],output_path,write=True)
                ut.showImage('Voronoi Diagram',img_voro_full,img_name[:-4],output_path,write=True)
                ut.showImage('Voronoi Diagram Segmented',img_voro_segmentation,img_name[:-4],output_path,write=True)
            
    
    
    

if __name__ == "__main__":
    main()


    





