# -- coding: utf-8 --
'''
demo: fit FLAME face model to 3D landmarks
Tianye Li <tianye.li@tuebingen.mpg.de>
'''

import numpy as np
import chumpy as ch
from os.path import join

from smpl_webuser.serialization import load_model
from fitting.landmarks import load_embedding, landmark_error_3d
from fitting.util import load_binary_pickle, write_simple_obj, safe_mkdir, mat_save,IglMatrixTonpArray
from facefit_lmk2d_strategy_3 import fit_lmk3d
import scipy.io as sio
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import myplot.vtkplot as vp
import quaternion
import sys
from configs.config import igl_python_path
sys.path.insert(0, igl_python_path)
import pyigl as igl
#dir(igl)
# -----------------------------------------------------------------------------





# -----------------------------------------------------------------------------

def run_fitting():
    # input landmarks
    '''
    lmk_path = './data/landmark_3d.pkl'
    lmk_3d = load_binary_pickle(lmk_path)

    q1 = quaternion.from_rotation_vector([0, 0, 0])
    #   quaternion.from_euler_angles(20,0,0)
    g_r = quaternion.as_rotation_matrix(q1)
    lmk_3d = np.asmatrix((lmk_3d + [0, 0.01, -0.01])) * np.asmatrix(g_r)
    lmk_3d = np.asarray(lmk_3d)
    print "loaded 3d landmark from:", lmk_path
    fig = plt.figure(figsize=plt.figaspect(1))
    ax = plt.subplot(111, projection='3d')
    ax.scatter(lmk_3d[:, 0], lmk_3d[:, 1], lmk_3d[:, 2], c='r')
    ax.set_zlabel('Z')
    ax.set_ylabel('Y')
    ax.set_xlabel('X')
    ax.set_xlim(ax.get_xlim()[::-1])
    plt.show()
    '''
    # model
    model_path = './models/generic_model.pkl'  # change to 'female_model.pkl' or 'generic_model.pkl', if needed
    model = load_model(
        model_path)  # the loaded model object is a 'chumpy' object, check https://github.com/mattloper/chumpy for details
    print "loaded model from:", model_path

    # landmark embedding
    lmk_emb_path = '../data/lmk_embedding_intraface_to_flame.pkl'
    lmk_face_idx, lmk_b_coords = load_embedding(lmk_emb_path)
    lmk_v = igl.eigen.MatrixXd()
    lmk_f =igl.eigen.MatrixXi()
    igl.readOBJ('C:/Users/hehua2015/Pictures/niutou/seg/maya_6.obj',lmk_v,lmk_f)
    lmk_v = IglMatrixTonpArray(lmk_v)
#    mat_save({'lmk_face_idx': lmk_face_idx}, 'lmk_face_idx.mat')
#    mat_save({'lmk_b_coord': lmk_b_coords}, 'lmk_b_coords.mat')
#    print "loaded lmk embedding"
    face_select_lmk = np.array([2210,1963,3486,3382,3385,3389,3392,3396,3400,3599,3594,3587,3581,3578,3757,568,728,
              3764, 3158, 335, 3705, 2178,
              673, 3863, 16, 2139, 3893,
              3553, 3561, 3501, 3564,
              2747, 2749, 3552, 1617, 1611,
              2428, 2383, 2493, 2488, 2292, 2335,
              1337, 1342, 1034, 1154, 959, 880,
              2712, 2850, 2811, 3543, 1694, 1735, 1576, 1770, 1801, 3511, 2904, 2878,
              2715, 2852, 3531, 1737, 1579,1793,3504,2896
              ])
    #逆时针 ，前到后
    body_select_lmk = np.array([3277,3240,3222,3341])
    target_face_lmk_v = lmk_v[face_select_lmk,0:2]
    target_body_lmk_v = lmk_v[body_select_lmk,0:2]
    # output
    output_dir = '../output'
    safe_mkdir(output_dir)

    # weights
    weights = {}
    weights['lmk'] = 1.0
    weights['shape'] = 0.001
    weights['expr'] = 0.001
    weights['pose'] = 0.1

    # optimization options
    import scipy.sparse as sp
    opt_options = {}
    opt_options['disp'] = 1
    opt_options['delta_0'] = 0.1
    opt_options['e_3'] = 1e-4
    opt_options['maxiter'] = 10
    sparse_solver = lambda A, x: sp.linalg.cg(A, x, maxiter=opt_options['maxiter'])[0]
    opt_options['sparse_solver'] = sparse_solver
    target_lmk_v = np.concatenate((target_face_lmk_v,target_body_lmk_v))

    landmark_v = igl.eigen.MatrixXd()
    landmark_f = igl.eigen.MatrixXi()
    igl.readOBJ('L:/yuanqing/imgs/imgs/vrn_result/niutou/01' + '/01_corr_landmark_cast_sym' + '.obj', landmark_v,
                landmark_f)
    landmark_v = np.array(landmark_v)
    landmark_body = np.array([[586,336,130],[562,369,150],[709,295,160],[727,262,150]])
    landmark_body[:,1] = 683-landmark_body[:,1]
#    landmark_body_f = np.array([[0,1,2],[0,2,3]])
#    igl.writeOBJ('L:/yuanqing/imgs/imgs/vrn_result/niutou/01' + '/01_landmark_body_plane' + '.obj', igl.eigen.MatrixXd(landmark_body.astype('float64')),
#                igl.eigen.MatrixXi(landmark_body_f.astype('intc')))

    target_lmk_v = np.concatenate((landmark_v[:,:], landmark_body[:,:]))

        # run fitting
    mesh_v, mesh_f, parms = fit_lmk3d(lmk_3d=target_lmk_v,  # input landmark 3d
                                      model=model,  # model
                                      mesh_faces=model.f,
                                      lmk_face_idx=lmk_face_idx,
                                      lmk_b_coords=lmk_b_coords,
                                      lmk_facevtx_idx=face_select_lmk, lmk_bodyvtx_idx=body_select_lmk,  # landmark embedding
                                      weights=weights,  # weights for the objectives
                                      shape_num=300, expr_num=100, opt_options=opt_options)  # options
    #    vp.trisurf(align_v, align_f, rendertype='wireframe')
    #    vp.scatter(pts=lmk_3d,alpha=1,mode='sphere',scale=0.001)
    # vp.trisurf(v_final, f_final, rendertype='wireframe', color3f=(0,0,0), alpha=0.3)
    #    vp.show()
    # write result
    output_dir = 'L:/yuanqing/imgs/imgs/vrn_result/niutou/01/'
    output_path = join(output_dir, 'fit_lmk3d_result_01.obj')
    write_simple_obj(mesh_v=mesh_v, mesh_f=mesh_f, filepath=output_path, verbose=False)
    return
    vp.trisurf(mesh_v, mesh_f, rendertype='wireframe')
    vp.scatter(pts=target_lmk_v, alpha=1, mode='sphere', scale=0.001)
    # vp.trisurf(v_final, f_final, rendertype='wireframe', color3f=(0,0,0), alpha=0.3)
    vp.show()
    write_simple_obj(mesh_v=mesh_v, mesh_f=mesh_f, filepath=output_path, verbose=False)


# -----------------------------------------------------------------------------

if __name__ == '__main__':
    run_fitting()

