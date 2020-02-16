import Utils as ut
import PreProcessing as pp

import itertools
import operator


#DA RIMUOVERE---------------------------------------------------
class Node:
    def __init__(self):
        self.name: str = ''
        self.children: List[Node] = []
        self.parent: Node = self

    def __getitem__(self, i: int) -> 'Node':
        return self.children[i]

    def add_child(self):
        child = Node()
        self.children.append(child)
        child.parent = self
        return child

    def __str__(self) -> str:
        def _get_character(x, left, right) -> str:
            if x < left:
                return '/'
            elif x >= right:
                return '\\'
            else:
                return '|'

        if len(self.children):
            children_lines: Sequence[List[str]] = list(map(lambda child: str(child).split('\n'), self.children))
            widths: Sequence[int] = list(map(lambda child_lines: len(child_lines[0]), children_lines))
            max_height: int = max(map(len, children_lines))
            total_width: int = sum(widths) + len(widths) - 1
            left: int = (total_width - len(self.name) + 1) // 2
            right: int = left + len(self.name)

            return '\n'.join((self.name.center(total_width),
                ' '.join(map(lambda width, position: _get_character(position - width // 2, left, right).center(width),
                             widths, itertools.accumulate(widths, operator.add))),
                *map(lambda row: ' '.join(map(
                        lambda child_lines: child_lines[row] if row < len(child_lines) else ' ' * len(child_lines[0]),
                        children_lines)),range(max_height))))
        else:
            return self.name
#----------------------------------------------------------------

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
    contours,_  = ut.cv.findContours(ut.np.uint8(ut.np.logical_not(binarized_img)),ut.cv.RETR_EXTERNAL,ut.cv.CHAIN_APPROX_SIMPLE) 
    points = []
    for contour in contours:
        M = ut.cv.moments(contour)
        # calculate x,y coordinate of center
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            points.append((cX,cY))
        else:
            cX, cY = 0, 0
            
        ut.cv.circle(output_img, (cX, cY), 5, (0, 255, 0), -1)
    points.sort(key=lambda x:x[0])
        
    return points

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
    return ut.kneighbors_graph(points, n_neighbors=k, mode='distance')

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
    full_tree = ut.minimum_spanning_tree(G, overwrite=True)

    return full_tree,ut.np.array(full_tree.nonzero()).T 

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
        if ut.rectContains(rect, pt1) and ut.rectContains(rect, pt2) and ut.rectContains(rect, pt3) :
            ut.cv.line(img, pt1, pt2, delaunay_color, 1, ut.cv.LINE_AA, 0)
            ut.cv.line(img, pt2, pt3, delaunay_color, 1, ut.cv.LINE_AA, 0)
            ut.cv.line(img, pt3, pt1, delaunay_color, 1, ut.cv.LINE_AA, 0)

def drawVoronoi(img, subdiv) :
    '''
    Draw the Voronoi Area diagram using Delaunay subdivision.
    
    Parameters
    ----------
    img: array of image pixels.
    subdiv : Delaunay subdivision of points (centroids)

    '''
    facets, _ = subdiv.getVoronoiFacetList([]) #get the Voronoi facet list 
    blank_img = ut.np.ones([img.shape[0],img.shape[1]],dtype=ut.np.uint8)*255
    
    
    #img2 = img.copy()
    voro_points = []
    for i in range(len(facets)) :
        ifacet_arr = []
        for f in facets[i] : #takes the numpy array of facet points and puts it in a new array
            ifacet_arr.append(f)
        ifacet = ut.np.array(ifacet_arr, ut.np.int) #makes the numpy array in a simple integer array
        ifacets = ut.np.array([ifacet])#builds an array of the previous integer array for make it compatible with the following method.
        height,width = img.shape[:2]
        x_arr = []
        y_arr = []
        area = ut.cv.contourArea(ifacets)
        print('area',area)
        if area>600:
            for i in range(len(ifacets[0])-1):
                x1,y1 = ifacets[0][i]
                x2,y2 = ifacets[0][i+1]
                #if(ut.euclidean_distance(ifacets[0][i],ifacets[0][i+1])<60):
                ut.cv.line(img, (x1, y1), (x2, y2), (0,0,255), 1, ut.cv.LINE_AA)
                ut.cv.line(blank_img, (x1, y1), (x2, y2), (0,0,255), 1, ut.cv.LINE_AA)
                #voro_points.append((x1,y1))
                        
            #if x1>=0 and x2>=0 and y1>=0 and y2>=0:
                #if x1<=width and x2<=width and y1<=height and y2<=height: 
                    #print(euclidean_distance(ifacets[0][i],ifacets[0][i+1]))
                    #if(euclidean_distance(ifacets[0][i],ifacets[0][i+1])<70):
                if x1>=0 and y1>=0 :
                    if x1<=width and y1<=height:        
                        voro_points.append((x1,y1))
                    
            
            x1,y1 = ifacets[0][len(ifacets[0])-1]
            #voro_points.append((x1,y1))
        '''
        if len(x_arr) != len(y_arr):
            print('AHAHAHAHAH')
        area = ut.polyArea(x_arr,y_arr)
        print('area', area)
        '''
    
        #cv.polylines(img2, ifacets, True, (255, 0, 0), 1, cv.LINE_AA, 0)#draws facet lines from the array of array ifacets
    #showImage('met 1',img)
    #showImage('met 2',img2)
    ut.showImage('White + voro', blank_img)
    return voro_points, ~blank_img

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
    subdiv = ut.cv.Subdiv2D(rect) #creates an empty Delaunay subdivision with the rectangle size
    
    for p in points : #add 2D points into Delaunay subdivisiom
        subdiv.insert(p)
        
    
    drawDelaunay(img, subdiv, (0, 0, 255)) #draw the triangulation using Delaunay subdivision.
    for p in points :
        ut.cv.circle(img, p, 2, (255,0,0), ut.cv.FILLED, ut.cv.LINE_AA, 0 )
    voro_points, voro_blank_inv = drawVoronoi(img_voronoi,subdiv) #draw the Voronoi diagram using Delaunay subdivision.
    
    ut.showImage('Delaunay Triangulation',img)
    ut.showImage('Voronoi Diagram',img_voronoi)
    
    return img, img_voronoi, voro_points, voro_blank_inv
    
    
def docstrum(input_img, output_img, edges, points, thresh_dist):
    ut.plt.figure(figsize=(30,20))
    ut.plt.imshow(input_img, 'gray')    
    for edge in edges:
        i,j = edge[2]
        distan = edge[1]
        
        if distan < thresh_dist:
            ut.cv.line(output_img, (points[i, 0], points[i, 1]), (points[j, 0], points[j, 1]), (0,0,0), 3, ut.cv.LINE_AA)
            ut.plt.plot([points[i, 0], points[j, 0]], [points[i, 1], points[j, 1]], c='r')
    ut.plt.show()    
    
def cutImage(image_bin,nPixel,space,verticalCut: bool = False):
    if verticalCut:
        image_bin = image_bin.T
        
    #Counting black pixels per row (axis=0: col, axis=1:row)
    counts,_ = ut.projection(image_bin)

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
        ut.cv.imwrite('verticalCut.tif', image_bin)
    else:
        ut.cv.imwrite('horizontalCut.tif', image_bin)
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
        #showImage('oo',crop_img)
        ut.cv.imwrite(path + new_imgname,crop_img)
        #cv2.imwrite(new_imgname, crop_img)
        #plt.imshow(crop_img, 'gray'),plt.show()
    tree = Node()
    tree.name = 'Page(root)'
    '''
    typeNode =[]
    label=[]
    
    XY_Tree.add_vertices(1)
    label.append('Pg')
    typeNode.append('root')
    
    XY_Tree_Node=0
    '''
    
    #vertical cut
    filelist = []
    print(path)
    path = path + new_imgname[:-21]
    print(path)
    for infile in ut.glob.glob (ut.os.path.join (path, 'prova*_horizCrop*.tif')):
        filelist.append (infile)
    filelist.sort()
    print(filelist)
    x=0
    y=0
    
    for file in filelist:
        newRead = ut.cv.imread(file, 0)
        inf = cutImage(newRead,4,21,True)#4,21
        if len(inf)==1:
            print('NoCut')
            title=input("Name: ")
            tree.add_child()
            tree[x].name = title
            ut.plt.title(title)
            ut.plt.imshow(newRead, 'gray'),ut.plt.axis('off'),ut.plt.show()
        else:
            flag=False
            for h in range(len(inf)-1):
                cropV = newRead[ 0:newRead.shape[0], inf[h][2]:inf[h+1][1] ]
                newNameV = file[:-4]+'_finalCut'+str(h)+'.tif'
                ut.cv.imwrite(newNameV, cropV)
                title=input("Name: ")
                #XY_Tree_Node += 1
                if len(inf)<=2:
                    #XY_Tree.add_vertices(1)
                    #label.append(title)
                    #typeNode.append('leaf')
                    #XY_Tree.add_edges ([(0,XY_Tree_Node)])
                    tree.add_child()
                    tree[x].name = title
                    x=x+1
                elif flag==False:
                    tree.add_child()
                    print('x,y',x,y)
                    tree[x].name = 'O'
                    tree[x].add_child()
                    tree[x][y].name = title
                    '''
                    XY_Tree.add_vertices(1)
                    label.append(' ')
                    typeNode.append('node')
                    XY_Tree.add_edges ([(0,XY_Tree_Node)])
                    XY_Tree.add_vertices(1)
                    label.append(title)
                    typeNode.append('leaf')
                    count = 0
                    XY_Tree.add_edges ([(XY_Tree_Node,XY_Tree_Node+1)])
                    '''
                    flag=True
                    #x=x+1
                    
                else:
                    tree[x-1].add_child()
                    tree[x-1][y].name = title
                    '''
                    XY_Tree.add_vertices(1)
                    label.append(title)
                    typeNode.append('leaf')
                    count += 1
                    XY_Tree_Node += 1
                    XY_Tree.add_edges ([(XY_Tree_Node-2*count,XY_Tree_Node-count+1)])
                    '''
                y=y+1
                flag=False
                ut.plt.title(title)
                ut.plt.imshow(cropV, 'gray'),ut.plt.axis('off'), ut.plt.show()
        y=0
    
    print(tree)
    #print(XY_Tree)
    #return pippo,typeNode,label
    #return typeNode,label
