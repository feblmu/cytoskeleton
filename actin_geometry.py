#!/usr/bin/pyhton

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.filters import uniform_filter
from scipy.ndimage import (binary_erosion, binary_dilation)
from scipy import ndimage
import copy

######################################################
# visulization

def show_image(imgs, alphas=1., cmaps=None, title='', 
               fig=None, pos=[0.1,0.1,0.8,0.8], cb_index=-1, cb_label=''):
    """
    imgs: list of images/arrays that get displayed one after the other
    alphas: list of alpha values corresponding to the images
    cmpas: list of colormaps for the images
    title: title of plot
    fig: figure where subplot can be added
    pos: position of subplot in figure
    cb_index: if colorbar is needed, the index of the corresponding images for which the colorbar is shown
    cb_label: label of colorbar
    return: None
    """
    if cmaps== None:
        cmaps = ['magma' for img in imgs]
    if alphas == 1.:
        alphas = [1. for img in imgs]
    
    if fig == None:
        fig = plt.figure(figsize=(3.,3.),dpi=200)
    ax=fig.add_axes(pos)
    
    for i, img in enumerate(imgs):   
        im = ax.imshow(img, cmap=cmaps[i], alpha=alphas[i])
        if i == cb_index:
            cbar = plt.colorbar(im, label=cb_label)
            cbar.ax.tick_params(labelsize=8) 
            cbar.set_label(label=cb_label, size=8)
    
    
    ax.text(0.5, 0.9, title, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, 
            color='white', fontsize=8, backgroundcolor='c')
    
    # style
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)    
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.show()
    
    
def show_geometry(ov_img, pixel_graph, mask, nodes, title='', 
               ms=2., 
               dpi=50,figsize=(16.,9.)):
    """
    ov_img: image of electron tomogram data
    pixel_graph: image of the topological backbone of the actin mesh
    mask: mask of extracelluar space
    nodes: dict containing nodes of topology with their position and type
    title: plot title
    ms: markersize (nodes)
    dpi: resolution of figure
    figsize: size of figure in inches
    return: None
    """

    fig = plt.figure(figsize=figsize,dpi=dpi)
    ax=fig.add_subplot(111)
    data = ax.imshow(ov_img, cmap='Greys')

    tmp = copy.copy(mask)
    tmp[np.where(tmp==1.)] = np.nan  
    mask_ax = ax.imshow(tmp, cmap='Paired', alpha=0.8)#, vmin=-0.5, vmax=1.)

    tmp = copy.copy(pixel_graph)
    tmp[np.where(tmp==0.)] = np.nan
    actin_geometry = ax.imshow(tmp, cmap='cividis', vmin=0., vmax=1.15)

    n_i = 0
    e_i = 0
    c_i = 0
    
    for node in nodes:
        tp = nodes[node]['type']
        com = nodes[node]['com']
        
        if tp == 'node':
            if n_i == 0:
                ax.plot(com[1], com[0], 'o', label='Node', markersize=ms, color='limegreen')
            else: 
                ax.plot(com[1], com[0], 'o', markersize=ms, color='limegreen')
            n_i = 1
        elif tp == 'end':
            if e_i == 0:
                ax.plot(com[1], com[0], 'o', label='End', markersize=ms, color='darkorchid')
            else: 
                ax.plot(com[1], com[0], 'o', markersize=ms, color='darkorchid')
            e_i = 1
        elif tp == 'cut':
            if c_i == 0:
                ax.plot(com[1], com[0], 'o', label='Cut', markersize=ms, color='cyan')
            else: 
                ax.plot(com[1], com[0], 'o', markersize=ms, color='cyan')
            c_i = 1


    ax.legend(fontsize=12, loc=(0.75,.8))
    
    ax.text(0.5, 0.97, title, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, 
            color='white', fontsize=16)
    # style
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)    
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.show()
        
    
##########################################################
# binarization of intracellular space into actin and background

def local_mean_filter(nd_arr, window_size):
    """
    compute local mean at every point in space of nd_arr 
    within a window with size: window_size 
    """
    return uniform_filter(nd_arr, size=window_size, mode='reflect')
    
def local_std_filter(nd_arr, window_size):
    """
    compute local standard deviation of sample at every point in space of nd_arr 
    within a window with size: window_size 
    """
    squared_mean = np.square(local_mean_filter(nd_arr, window_size))
    mean_squared = uniform_filter(np.square(nd_arr), size=window_size, mode='reflect')
    sample_std_corr = np.square(window_size)/(np.square(window_size)-1.)
    std = np.sqrt( sample_std_corr *  (mean_squared - squared_mean) )
    return std

def threshold_niblack(image, window_size, k, q):
    """
    compute local threshold for segmentation using niblack's method
    window_size: window to compute mean and standard dev
    threshold is given by: T = (mean + k * std) * q
    """
    return (local_mean_filter(image, window_size) - k* local_std_filter(image, window_size)) * q

def threshold_sauvola(image, window_size, k, r=None):
    """
    compute local threshold for segmentation using sauvolas's method
    window_size: window to compute mean and standard dev
    threshold is given by: T = mean * (1 + k * (std / r - 1) )
    remark: use negative k values as light parts should be segmented out
    """
    if r==None:
        r = np.zeros(np.shape(image))
        r[:,:] = np.mean(image.flatten()) * 2.
    else:
        if np.shape(image) != np.shape(r):
            raise AssertionError('r and image have to have same shape.')
   
    return ( local_mean_filter(image, window_size) * (1. + k * (local_std_filter(image, window_size)/r - 1.)))
    
##########################################
# find membrane mask to remove so far unmasked parts of the membrane

def membrane_voxel_density(boundary_mask, boundary_region, window_size):
    """
    compute density of membrane voxels in boundary region
    """
    density = local_mean_filter(boundary_region, 
                                window_size=window_size) / local_mean_filter(boundary_mask, window_size=window_size)
    density[np.where(boundary_region==0.)]=0.
    return density

def compute_membrane_mask(mask_dst_trans, segmented_img,voxel_size=1.,
                  max_boundary_distance=[2,4,6,8], membrane_cutoff=0.8, smoothing=True):
    """
    identify voxels in boundary region that are part of the membrane and create a mask for those membrane voxels
    this method makes use of the fact that regions close to the boundary that contain membrane voxels lead to a higher voxels density
    i.e. the mean value within a local window around each voxel in the binarized dataset is higher if sections of the mebrane are 
    unmasked
    
    mask_dst_trans: distance transform applied to mask; distance to boundary in intracellular space
    segmented_img: binarized dataset
    voxel_size: size of the voxel in nm or angström depending on dataset 
    max_boundary_distance: list of values representing the a distence to the boundary (in the distance transfrom)
                           last value of list should be the max. distance within which voxels belonging to the membrane are suspected
                           use more steps with additional smaller values to make detection more reliable
    membrane_cutoff: density cutoff value, if a local mean value in the binarized data is above this value 
                     voxel gets identified as membrane voxel
    """
    
    # references
    distance_transform = mask_dst_trans
    segmented_data = segmented_img
    
    # compute physical distance from pixel number distance
    distance_transform *= voxel_size
    
    # sort boundary distances from largest to smallest value
    max_boundary_distance = np.flip(np.sort(max_boundary_distance))
    
    #find voxels that belong to the membrane
    membrane_mask = np.zeros(np.shape(segmented_data))
    for bnd_dis in max_boundary_distance:
        
        # find region that is within bnd_dis (Angström) range to boundary of manually segmented data
        boundary_mask = np.zeros(np.shape(segmented_data))    
        boundary_indices = np.logical_and(distance_transform>0, distance_transform<bnd_dis)
        boundary_mask[boundary_indices] = 1.
        
        # get all voxels in segemted data in boundary region
        boundary_region = np.zeros(np.shape(segmented_data))
        boundary_region[boundary_indices]  = segmented_data[boundary_indices]
    
        # compute density of segmented voxels in boundary region if density is high then voxels are presumably
        # part of the membrane
        voxel_density = membrane_voxel_density(boundary_mask, boundary_region, window_size=21)
        
        # decide by a cutoff value of voxel is mambrane or actin
        voxel_density[np.where(voxel_density>membrane_cutoff)] = 1.  # membrane
        voxel_density[np.where(voxel_density<=membrane_cutoff)] = 0.  # not membrane
        
        # merge membrane masks for different sizes of boundary region
        membrane_mask=np.logical_or(membrane_mask, voxel_density) 
    
    if smoothing == True:
        struct = np.ones([3,3])
        membrane_mask = binary_dilation(membrane_mask, structure=struct)
     
    # convert membrane mask from bool to float
    membrane_mask=membrane_mask.astype(float)
    
    return membrane_mask
    
##########################################
# reduce binarized data to lines from which a graph can be extracted

def index_distance_transform_2d(dst_data):
    """
    sort all pixels of an image with respect to their distance transfrom 
    value
    start with smallest values
    """
    
    # array of unique values that are contained in distance_values array
    unique_dst_vals = np.unique(dst_data)
    unique_dst_vals = unique_dst_vals[1:] # remove zero as it is background
 
    actin_indices = np.array(np.where(dst_data == unique_dst_vals[0]))
    actin_indices = np.transpose(actin_indices)

    for dst in unique_dst_vals[1:]:
        tmp = np.array(np.where(dst_data == dst))
        tmp = np.transpose(tmp)
        actin_indices = np.concatenate((actin_indices, tmp))   
    
    nonzeros, tmp = np.shape(actin_indices)
    print(str(nonzeros) + " voxels indexed in distance transform")
    return actin_indices

def find_pixel_graph_2d(dst_data_in, structure, n_local, repeat=True,):
    """
    thinning algorithm applied to a binarized image that preserves the topolgy.
    The voxels are sorted by their distance transform value. Voxels with lowes distancd transform are subsequently set to zero based     on the following topology-preserving rules: voxels were kept if their removal would "break’’ a filament or dissociate a filament from the volume boundary, or if the voxel lay at the end of a filament branch. This procedure yielded lines of connected voxels representing the structure of the actin cytoskeleton
    dst_data_in: distance transform applied to binarized images
    strcuture: connectivity between voxels
    n_local: window_size to check topology
    repeat: if true returns the number of removed pixels, so that the loop can end if no more pixels get removed
    """
     
    label_data, nb_labels = ndimage.label(dst_data_in, structure=structure)
    print('global number of labels in distance transform of filaments before extracting pixel graph: ',nb_labels)
    
    # get shape of distance transform data
    nx, ny = np.shape(dst_data_in)
    
    actin_indices = index_distance_transform_2d(dst_data_in)
    
    # add frame to distance transform data
    # inner pixel of frame is used to fix cut filaments at the ends of the data space so that the proceture below will not remove them
    # rest of frame can be filled with 0
    nx, ny = np.shape(dst_data_in)
    dst_data = np.zeros([nx+2*n_local, ny+2*n_local])
    dst_data[n_local:-n_local, n_local:-n_local]=copy.deepcopy(dst_data_in)
    dst_data[n_local-1,n_local:-n_local] = dst_data[n_local,n_local:-n_local]
    dst_data[-n_local,n_local:-n_local] = dst_data[-n_local-1,n_local:-n_local]
    dst_data[n_local:-n_local,n_local-1] = dst_data[n_local:-n_local,n_local]
    dst_data[n_local:-n_local,-n_local] = dst_data[n_local:-n_local,-n_local-1]

    # increase indices by +n_local (accout for frame)
    # and copy it to not change original array
    actin_indices =copy.deepcopy(actin_indices)+n_local

    N_not_removed = 0
    N_removed = 0

    nonzeros, tmp = np.shape(actin_indices) 
    for i in range(nonzeros):  # loop over all filament voxels
        # set remove True
        # if this will not be changes below voxel gets removed from final pixel graph
        remove = True
        
        # 0. test if voxel is part of surface dilation if yes do not remove
        if dst_data[actin_indices[i][0], actin_indices[i][1]] == 1000.:
            remove = False
        
        # 1. Test if removing voxel seperates a connected domain, if yes do not remove
        
        # cut local region from data around voxel that gets tested if it can be removed
        local_region = copy.copy(dst_data[actin_indices[i][0]-n_local:actin_indices[i][0]+n_local+1, 
                                          actin_indices[i][1]-n_local:actin_indices[i][1]+n_local+1])
        
        # count seperate regions in local environment
        label_local, nb_labels_local = ndimage.label(local_region, structure=structure)

        # set voxel to be removed to 0
        local_region[n_local, n_local] = 0.
        
        # if removing pixel seperates a connected region do not remove
        # -> do not disconnected regions of the same filament
        # therefor compure seperable regions in local region after voxel was set to 0
        label_tmp, nb_labels_tmp = ndimage.label(local_region, structure=structure)

        if nb_labels_tmp != nb_labels_local:
            remove = False  # do not remove pixel if number of labels has increased as it seperates a connected domain
        
        # 2. test if pixel is the end of a filement , if yes do not remove
        
        # get surrounding pixels
        neighbours = dst_data[actin_indices[i][0]-1:actin_indices[i][0]+2, 
                              actin_indices[i][1]-1:actin_indices[i][1]+2] *structure
        
        # count number of surrounding pixels that are zero i.e. are background and not filaments
        n_nonzeros = np.sum(neighbours.flatten()!=0.)
        
        # if 2 voxels are non-zero 
        # this means that all but one neighbours are background 
        # do not remove pixel as it is end tip of filament 
        if n_nonzeros <= 2:
            remove = False
        
        # 3. remove
        # if voxel is not end of filament or important for a persistent connection 
        if remove == True:
            dst_data[actin_indices[i][0], actin_indices[i][1]] = 0.
            N_removed += 1
            
        else:
            N_not_removed += 1
            

        if i % 1000 == 0:
            print(str(i) + " of " + str(nonzeros) + " voxels checked...")
        
    
        
    # remove frame again     
    dst_data = dst_data[n_local:-n_local, n_local:-n_local]
    
    # 
    label_data, nb_labels = ndimage.label(dst_data, structure=structure)
    
    print(str(N_removed) + " voxels removed.")
    print(str(N_not_removed) + " voxel remain in voxel graph")
    print('gloabal nuber of labels in pixel graph: ',nb_labels)
    
    if repeat == True:  # return N_removed to check if function should be repeated    
        return dst_data, N_removed
    else:
        return dst_data

    
    
def remove_small_fragments(binary_pixel_graph, structure, min_fragment_size=5):  
    
    """
    remove small fragemnts that are smaller than min_fragement_size and not connected to any other filaments
    
    structure. connectivity between pixels
    binary_pixel_graph: topological backbone, results of thinning algorithm
    return: binary_pixel_graph with fragments removed
    """
    label_data, nb_labels = ndimage.label(binary_pixel_graph, structure=structure)
    for label in range(1, nb_labels + 1):
        fragment = np.where(label_data == label)
        tmp, fragment_size = np.shape(fragment)
        if fragment_size < min_fragment_size:
            binary_pixel_graph[fragment] = 0.
    return binary_pixel_graph
   
    
def remove_small_end_tips(binary_pixel_graph, structure, min_end_length=5):
    """
    remove branches in pixel graph that consits of only one end voxel
    
    pixel_graph: extracted pixel graph containing only 1 and 0
    structure: defines connectivity between neighbouring voxels
    
    return: number of removed branches
    """
    
    graph_copy = segment_pixel_graph(binary_pixel_graph, structure=structure)

    # remove nodes 
    graph_copy[np.where(graph_copy == 2)] = 0.
    
    # find all domains in data with size 1
    labelled, nb = ndimage.label(graph_copy, structure=structure)
    
    removed = 0
    for i in range(nb):
        n_v = np.where(labelled == i)
        tmp, size = np.shape(n_v)
        tmp_pixels = graph_copy[n_v]
        
        # test if branch is connected to end node
        if 3 in tmp_pixels:
            if size < min_end_length:
                binary_pixel_graph[n_v] = 0.
                removed += 1 

    print('small end branches removed: ',removed)
    return binary_pixel_graph, removed

def segment_pixel_graph(binary_pixel_graph, structure):
    """
    SEGMENT VOXELS IN DIFFERENT CLASSES: node, cut, end & branch
    A voxel connected to exactly two other voxels was considered to be part of a branch. A voxel connected to more than two other voxels was classified as a node (branching point). Voxels connected to only one other voxel were classified as branch ends.
    
    pixel_graph: pixel graph strucutre containing only 1 and 0
    structure: defines connectivity between neighbouring voxels
    
    return: binary_pixel_graph with every pixel classified as node, cut, branch or background
    """
    
    for value in np.unique(binary_pixel_graph):
        if value not in [0., 1.]:
            print(np.unique(binary_pixel_graph))
            raise Exception('input pixel graph may only contain ones and zeros')
    
    # copy 
    graph_copy = copy.copy(binary_pixel_graph)   

    # FIND NODES AND END TIPS
    bin_pixel_indices = np.transpose(np.where(graph_copy==1.))

    ### nx, ny, nz = np.shape(binary_pixel_graph)
    nx, ny = np.shape(graph_copy)

    node_indices = []
    end_indices = []
    cut_indices = []

    # loop over all voxels of graph
    for i in bin_pixel_indices:

        ix = i[0]
        iy = i[1]
        ### iz = i[2]
        
        # test if voxel is at end of volume -> filament was cut
        
        ###if (ix == 0) or (ix == nx-1) or (iy == 0) or (iy == ny-1) or (iz == 0) or (iz == nz-1):
        if (ix == 0) or (ix == nx-1) or (iy == 0) or (iy == ny-1):
            
            ### cut_indices.append((ix, iy, iz))
            cut_indices.append((ix, iy))
        else:
            # slice out nearest neighbours of voxel
            # multiply by connectivty structure that was used for extraction of graph
            
            ### neighbourhood =  binary_pixel_graph[ix-1:ix+2, iy-1:iy+2, iz-1:iz+2] * structure
            neighbourhood =  graph_copy[ix-1:ix+2, iy-1:iy+2] * structure
            
            # count neares neighbours of voxel
            # subtract -1 as voxel can not be a neighbour of itself
            neighbours = np.nansum(neighbourhood) - 1
            
            if neighbours == 1:
                ### end_indices.append((ix, iy, iz))
                end_indices.append((ix, iy))
            if neighbours == 2:
                pass   # branch voxel
            if neighbours > 2: 
                ### node_indices.append((ix,iy, iz))
                node_indices.append((ix,iy))
            
    
    # color binary pixel graph
    for node in node_indices:
        ### graph_copy[node[0], node[1], node[2]] = 2.
        graph_copy[node[0], node[1]] = 2.
    for end in end_indices:
        ### graph_copy[end[0], end[1], end[2]]= 3.
        graph_copy[end[0], end[1]]= 3.
    for cut in cut_indices: 
        ### graph_copy[cut[0], cut[1], cut[2]] = 4.
        graph_copy[cut[0], cut[1]] = 4.
    
    return graph_copy
    
    
def identify_nodes(segmented_pixel_graph, structure):
    """
    create a dictonary with all nodes that contains the center of mass and the type of the node as additional information
    
    pixel_graph: array with every pixel classified as node, cut, branch or background
    structure: defines connectivity between neighbouring voxels
    
    return: dict containing all nodes with com and type
    """
    shape = np.shape(segmented_pixel_graph)
    # find node voxel that belong to the same node
    # use ndimage.label to find connected nodes
    nodes = np.zeros(shape)
    nodes[np.where(segmented_pixel_graph== 2.)] = 1.
    labelled_nodes, nb_nodes = ndimage.label(nodes, structure=structure)
    print('number of nodes:', nb_nodes)    
    # for every node compute center of mass
    com_nodes =ndimage.center_of_mass(nodes, labelled_nodes, range(1,nb_nodes+1))
    
    # find end voxels that belong to the same end
    ends = np.zeros(shape)
    ends[np.where(segmented_pixel_graph==3.)] = 1.
    labelled_ends , nb_ends = ndimage.label(ends, structure=structure)
    # find center of mass of ends
    com_ends =ndimage.center_of_mass(ends, labelled_ends, range(1,nb_ends+1))
    print('number of ends: ', nb_ends)
    
    # find cut voxels that belong to the same cut
    # use ndimage.label to finde connected cutvoxels
    cuts = np.zeros(shape)
    cuts[np.where(segmented_pixel_graph==4.)] = 1.
    labelled_cuts , nb_cuts = ndimage.label(cuts, structure=structure)
    com_cuts =ndimage.center_of_mass(cuts, labelled_cuts, range(1,nb_cuts+1))
    print('number of cuts: ', nb_cuts)
    
    # put all nodes, ends and cuts in a dict including type, center of mass,...
    nodes_dict = {}
    for i in range(nb_nodes):
        nodes_dict.update({str(i+1): {
            "com": com_nodes[i],
            "type": "node",
        }})
    for i in range(nb_ends):
        nodes_dict.update({str(i+1+nb_nodes): {
            "com": com_ends[i],
            "type": "end",
        }})
    for i in range(nb_cuts):
        nodes_dict.update({str(i+1+nb_nodes+nb_ends): {
            "com": com_cuts[i],
            "type": "cut",
        }})
   
    return nodes_dict

