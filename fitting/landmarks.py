# -- coding: utf-8 --
'''
Util funcitons for landmarks
Tianye Li <tianye.li@tuebingen.mpg.de>
'''

import numpy as np
import chumpy as ch
import cPickle as pickle

from fitting.util import load_binary_pickle,load_pickle

# -----------------------------------------------------------------------------

def load_embedding( file_path ):
    """ funciton: load landmark embedding, in terms of face indices and barycentric coordinates for corresponding landmarks
    note: the included example is corresponding to CMU IntraFace 49-point landmark format.
    """
    lmk_indexes_dict = load_pickle( file_path )
    lmk_face_idx = lmk_indexes_dict[ 'lmk_face_idx' ].astype( np.uint32 )
    lmk_b_coords = lmk_indexes_dict[ 'lmk_b_coords' ]
    return lmk_face_idx, lmk_b_coords

# -----------------------------------------------------------------------------

def mesh_points_by_barycentric_coordinates( mesh_verts, mesh_faces, lmk_face_idx, lmk_b_coords ):
    """ function: evaluation 3d points given mesh and landmark embedding
    """
    dif1 = ch.vstack([(mesh_verts[mesh_faces[lmk_face_idx], 0] * lmk_b_coords).sum(axis=1),
                    (mesh_verts[mesh_faces[lmk_face_idx], 1] * lmk_b_coords).sum(axis=1),
                    (mesh_verts[mesh_faces[lmk_face_idx], 2] * lmk_b_coords).sum(axis=1)]).T
    return dif1

# -----------------------------------------------------------------------------

def landmark_error_3d( mesh_verts, mesh_faces, lmk_3d, lmk_face_idx, lmk_b_coords, weight=1.0 ):
    """ function: 3d landmark error objective
    """

    # select corresponding vertices
    v_selected = mesh_points_by_barycentric_coordinates( mesh_verts, mesh_faces, lmk_face_idx, lmk_b_coords )
    lmk_num  = lmk_face_idx.shape[0]

    # an index to select which landmark to use
    lmk_selection = np.arange(0,lmk_num).ravel() # use all

    # residual vectors
    lmk3d_obj = weight * ( v_selected[lmk_selection] - lmk_3d[lmk_selection] )

    return lmk3d_obj
def landmark_error_2_3d( scale ,trans_2_3d, mesh_verts, target_lmk_3d_face, target_lmk_3d_body, lmk_facevtx_idx, lmk_bodyvtx_idx, face_weight=1.0,body_weight =1.0 ):
    """ function: 3d landmark error objective
    """
    # select corresponding vertices

    source_face_lmkvtx = mesh_verts[lmk_facevtx_idx]
    source_body_lmkvtx = mesh_verts[lmk_bodyvtx_idx]
    '''
    target = ch.concatenate((target_lmk_3d_face,target_lmk_3d_body))
    source = ch.concatenate((source_face_lmkvtx,source_body_lmkvtx))
    # residual vectors
    lmk3d_obj = (target - source)
    lmk3d_obj[0:target_lmk_3d_face.shape[0],:] = face_weight*lmk3d_obj[0:target_lmk_3d_face.shape[0],:]
    lmk3d_obj[target_lmk_3d_face.shape[0]:target_lmk_3d_face.shape[0]+target_lmk_3d_body.shape[0],:] = body_weight * lmk3d_obj[target_lmk_3d_face.shape[0]:target_lmk_3d_face.shape[0]+target_lmk_3d_body.shape[0],:]
    '''
    if( target_lmk_3d_face.shape[1] == 2):
        cast_source_face_lmkvtx = scale*source_face_lmkvtx[:,0:2]+trans_2_3d[0:2]
        cast_source_body_lmkvtx = scale*source_body_lmkvtx[:,0:2]+trans_2_3d[0:2]
        lmk3d_obj_face = face_weight * (target_lmk_3d_face - cast_source_face_lmkvtx)
        lmk3d_obj_body = body_weight * (target_lmk_3d_body - cast_source_body_lmkvtx)
    elif target_lmk_3d_face.shape[1] == 3:
        cast_source_face_lmkvtx = scale * source_face_lmkvtx[:,:] + trans_2_3d[:]
        cast_source_body_lmkvtx = scale * source_body_lmkvtx[:, 0:2] + trans_2_3d[0:2]
        lmk3d_obj_face = face_weight * (target_lmk_3d_face - cast_source_face_lmkvtx)
        lmk3d_obj_body = body_weight * (target_lmk_3d_body[:,0:2] - cast_source_body_lmkvtx)
    else:
        pass
    return lmk3d_obj_face,lmk3d_obj_body
def landmark_error_3d( scale ,trans_2_3d, mesh_verts,mesh_faces, target_lmk_3d_face, target_lmk_3d_body,lmk_face_idx, lmk_b_coords, lmk_facevtx_idx, lmk_bodyvtx_idx, face_weight=1.0,body_weight =1.0,use_lunkuo =False,use_3d_landmark =False,use_landmark_idx =None ):
    """ function: 3d landmark error objective
    """
    # select corresponding vertices
    v_selected = mesh_points_by_barycentric_coordinates( mesh_verts, mesh_faces, lmk_face_idx, lmk_b_coords )
    source_face_lmkvtx = mesh_verts[lmk_facevtx_idx[0:17]]
#    lmk_num  = lmk_face_idx.shape[0]
    source_body_lmkvtx = mesh_verts[lmk_bodyvtx_idx]
    # an index to select which landmark to use
#    lmk_selection = np.arange(0,lmk_num).ravel() # use all
    if use_lunkuo:
        frame_landmark_idx = range(0,17)+range(17, 60) + range(61, 64) + range(65, 68)
    else:
        frame_landmark_idx = range(17, 60) + range(61, 64) + range(65, 68)
    if use_landmark_idx == None:
        target_lmk_3d_face = target_lmk_3d_face[frame_landmark_idx,:]
    else:
        target_lmk_3d_face = target_lmk_3d_face[use_landmark_idx,:]
    if use_lunkuo:
        v_selected_merge = ch.vstack([source_face_lmkvtx,v_selected])
    else:
        v_selected_merge = v_selected
    if use_landmark_idx !=None:
        v_selected_merge = v_selected_merge[use_landmark_idx]
    # residual vectors
    if use_3d_landmark:
        pass
    else:
        target_lmk_3d_face = target_lmk_3d_face[:,0:2]
    if( target_lmk_3d_face.shape[1] == 2):
        cast_source_face_lmkvtx = scale*v_selected_merge[:,0:2]+trans_2_3d[0:2]
        cast_source_body_lmkvtx = scale*source_body_lmkvtx[:,0:2]+trans_2_3d[0:2]
        lmk3d_obj_face = face_weight * (target_lmk_3d_face[:,0:2] - cast_source_face_lmkvtx)
        lmk3d_obj_body = body_weight * (target_lmk_3d_body[:,0:2] - cast_source_body_lmkvtx)
    elif target_lmk_3d_face.shape[1] == 3:
        cast_source_face_lmkvtx = scale * v_selected_merge[:,:] + trans_2_3d[:]
        cast_source_body_lmkvtx = scale * source_body_lmkvtx[:, 0:2] + trans_2_3d[0:2]
        lmk3d_obj_face = face_weight * (target_lmk_3d_face - cast_source_face_lmkvtx)
        lmk3d_obj_body = body_weight * (target_lmk_3d_body[:,0:2] - cast_source_body_lmkvtx)
    else:
        pass

    return lmk3d_obj_face,lmk3d_obj_body


def landmark_error_3d_only( scale ,trans_3d, mesh_verts,mesh_faces,lmk_face_idx, lmk_b_coords,target_v,targrt_face,target_lmk_idx,weight):
    v_selected = mesh_points_by_barycentric_coordinates(mesh_verts, mesh_faces, lmk_face_idx, lmk_b_coords)

    source_lmkvtx = scale * v_selected[:, :] + trans_3d[:]
    target_lmk = target_v[target_lmk_idx, :]
    lmk3d_error = weight * (target_lmk - source_lmkvtx)
    return lmk3d_error

from sklearn.neighbors import NearestNeighbors
from time import time
from math import pi,cos

def p2perror(
            scale,trans_2_3d,mesh_verts,mesh_faces,target_3d_v,target_3d_f,mask_facevtx_idx,p2p_weight
            ):
        from fitting.util import get_vertex_normal
        source_face_lmkvtx = mesh_verts[mask_facevtx_idx]
        cast_source_face_lmkvtx = scale * source_face_lmkvtx[:, :] + trans_2_3d[:]
        cast_source_face_lmkvtx_np = cast_source_face_lmkvtx.r
        neigh = NearestNeighbors(n_neighbors=1)
        timer_start = time()
        neigh.fit(target_3d_v)
        timer_end = time()
        print 'neigh.fit'
        print "in %f sec\n" % (timer_end - timer_start)
        distances, indices = neigh.kneighbors(cast_source_face_lmkvtx_np, return_distance=True)
        source_normals = get_vertex_normal(mesh_verts.r,mesh_faces)
        target_normals = get_vertex_normal(target_3d_v, target_3d_f)
        select_source_idx = []
        select_target_idx = []
        for i in range(0,cast_source_face_lmkvtx_np.shape[0]):
            if i >3930:
                break
            source_idx = i
            source_vtx = cast_source_face_lmkvtx_np[i]
            source_normal = source_normals[i]
            target_idx = indices[i, 0]
            target_vtx = target_3d_v[target_idx]
            target_normal = target_normals[target_idx]
            if np.dot(source_normal,target_normal) > cos(pi /3):
                if np.linalg.norm(source_vtx - target_vtx) <0.04:
                    dir =(source_vtx - target_vtx)/np.linalg.norm(source_vtx - target_vtx)
                    if abs(np.dot(dir,source_normal))> cos(pi /3):
                        select_source_idx.append(source_idx)
                        select_target_idx.append(target_idx)
                        pass
        print 'len select_source_idx',len(select_source_idx),'len select_target_idx',len(select_target_idx)
        #target_p2p = target_3d_f[indices[:,0]]
        #使用点到面距离
        #lmk3d_p2p_face = p2p_weight * ((target_3d_v[select_target_idx] - cast_source_face_lmkvtx[select_source_idx])*source_normals[select_source_idx])
        lmk3d_p2p_face = p2p_weight * ((target_3d_v[select_target_idx] - cast_source_face_lmkvtx[select_source_idx]))
        return lmk3d_p2p_face