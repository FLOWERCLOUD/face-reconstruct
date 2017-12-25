'''
Util funcitons for landmarks
Tianye Li <tianye.li@tuebingen.mpg.de>
'''

import numpy as np
import chumpy as ch
import cPickle as pickle

from fitting.util import load_binary_pickle

# -----------------------------------------------------------------------------

def load_embedding( file_path ):
    """ funciton: load landmark embedding, in terms of face indices and barycentric coordinates for corresponding landmarks
    note: the included example is corresponding to CMU IntraFace 49-point landmark format.
    """
    lmk_indexes_dict = load_binary_pickle( file_path )
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
def landmark_error_3d( scale ,trans_2d, mesh_verts, target_lmk_3d_face, target_lmk_3d_body, lmk_facevtx_idx, lmk_bodyvtx_idx, face_weight=1.0,body_weight =1.0 ):
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
        cast_source_face_lmkvtx = scale*source_face_lmkvtx[:,0:2]+trans_2d
        cast_source_body_lmkvtx = scale*source_body_lmkvtx[:,0:2]+trans_2d
        lmk3d_obj_face = face_weight * (target_lmk_3d_face - cast_source_face_lmkvtx)
        lmk3d_obj_body = body_weight * (target_lmk_3d_body - cast_source_body_lmkvtx)
    elif target_lmk_3d_face.shape[1] == 3:
        lmk3d_obj_face = face_weight * (target_lmk_3d_face - source_face_lmkvtx)
        lmk3d_obj_body = body_weight * (target_lmk_3d_body - source_body_lmkvtx)
    else:
        pass
    return ch.vstack([lmk3d_obj_face,lmk3d_obj_body])