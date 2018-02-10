# -- coding: utf-8 --
import numpy as np
import chumpy as ch
from os.path import join

from smpl_webuser.serialization import load_model
from fitting.landmarks import load_embedding, landmark_error_3d
from fitting.util import load_binary_pickle, write_simple_obj, safe_mkdir, mat_save,IglMatrixTonpArray
import scipy.io as sio
import quaternion
import sys
sys.path.insert(0, "D:/mprojects/libiglfull/libigl/python/build/x64/Release")
import pyigl as igl



def fit_lmk3d_old(lmk_3d,  # input landmark 3d
              model,  # model
              lmk_facevtx_idx, lmk_bodyvtx_idx,  # landmark embedding
              weights,  # weights for the objectives
              shape_num=300, expr_num=100, opt_options=None):
    """ function: fit FLAME model to 3d landmarks

    input:
        lmk_3d: input landmark 3d, in shape (N,3)
        model: FLAME face model
        lmk_face_idx, lmk_b_coords: landmark embedding, in face indices and barycentric coordinates
        weights: weights for each objective
        shape_num, expr_num: numbers of shape and expression compoenents used
        opt_options: optimizaton options

    output:
        model.r: fitted result vertices
        model.f: fitted result triangulations (fixed in this code)
        parms: fitted model parameters

    """

    # variables
    shape_idx = np.arange(0, min(300, shape_num))  # valid shape component range in "betas": 0-299
    expr_idx = np.arange(300, 300 + min(100, expr_num))  # valid expression component range in "betas": 300-399
    used_idx = np.union1d(shape_idx, expr_idx)
    model.betas[:] = np.random.rand(model.betas.size) * 0.0  # initialized to zero
    model.pose[:] = np.random.rand(model.pose.size) * 0.0  # initialized to zero
    # free_variables = [ model.trans, model.pose, model.betas[used_idx] ]
    free_variables = [model.pose, model.betas[used_idx]]
    # weights
    print "fit_lmk3d(): use the following weights:"
    for kk in weights.keys():
        print "fit_lmk3d(): weights['%s'] = %f" % (kk, weights[kk])

        # objectives
    # lmk
    scale = ch.array([1])

    # options
    if opt_options is None:
        print "fit_lmk3d(): no 'opt_options' provided, use default settings."
        import scipy.sparse as sp
        opt_options = {}
        opt_options['disp'] = 1
        opt_options['delta_0'] = 0.1
        opt_options['e_3'] = 1e-4
        opt_options['maxiter'] = 100
        sparse_solver = lambda A, x: sp.linalg.cg(A, x, maxiter=opt_options['maxiter'])[0]
        opt_options['sparse_solver'] = sparse_solver

    # on_step callback
    def on_step(_):
        pass

    # optimize
    # step 1: rigid alignment
    from time import time
    timer_start = time()
    '''

    print "\nstep 1: start rigid fitting..."
    ch.minimize( fun      = lmk_err,
                 x0       = [ model.trans, model.pose[0:3],scale ],
                 method   = 'dogleg',
                 callback = on_step,
                 options  = opt_options )
    timer_end = time()
    align_v = scale*model.r
    align_f = model.f
    print "step 1: fitting done, in %f sec\n" % ( timer_end - timer_start )
    print "scale： %f\n" % scale
    print "model.trans"
    print model.trans
    print "model.rot"
    print model.pose[0:3]
    '''
    # optimize
    # step 1: rigid alignment

    g_scale = ch.array([1000])
    g_trans_2d = ch.array([0,0])
    print "\nstep 1: update global pose..."
    lmk_err = landmark_error_3d(
                                scale = g_scale,
                                trans_2d = g_trans_2d,
                                mesh_verts=model,
                                target_lmk_3d_face=lmk_3d[0:68,:],
                                target_lmk_3d_body=lmk_3d[68:72,:],
                                lmk_facevtx_idx=lmk_facevtx_idx,
                                lmk_bodyvtx_idx=lmk_bodyvtx_idx,
                                face_weight=0,
                                body_weight= 1
                                )
    # regularizer
    shape_err = weights['shape'] * model.betas[shape_idx]
    expr_err = weights['expr'] * model.betas[expr_idx]
    pose_err = weights['pose'] * model.pose[3:6]  # exclude global rotation
    objectives = {}
    objectives.update({'lmk': lmk_err, 'shape': shape_err, 'expr': expr_err, 'pose': pose_err})
    objectives_pose = {}
    objectives_pose.update({'lmk': lmk_err})
    ch.minimize( fun      = objectives_pose,
                 x0       = [ model.trans,g_trans_2d,model.pose[0:3]],
                 method   = 'dogleg',
                 callback = on_step,
                 options  = opt_options )
    timer_end = time()
    print "step 1: global pose fitting done, in %f sec\n" % ( timer_end - timer_start )
    print "model.trans"
    print model.trans
    print "head.rot"
    print model.pose[3:6]
    print "all pose"
    print model.pose[:]
    print "g_scale"
    print g_scale
    print "g_trans_2d"
    print g_trans_2d
    output_dir = 'L:/yuanqing/imgs/imgs/vrn_result/niutou/01/'
    output_path = join(output_dir, 'step_1.obj')
    g_v =  g_scale.r*model.r[:,:]
    g_v[:,0:2] = g_v[:,0:2]+g_trans_2d.r
    write_simple_obj(mesh_v=g_v, mesh_f=model.f, filepath=output_path, verbose=False)
    # step 2: update head pose
    lmk_err = landmark_error_3d(mesh_verts=model,
                                scale=g_scale,
                                trans_2d=g_trans_2d,
                                target_lmk_3d_face=lmk_3d[0:68,:],
                                target_lmk_3d_body=lmk_3d[68:72,:],
                                lmk_facevtx_idx=lmk_facevtx_idx,
                                lmk_bodyvtx_idx=lmk_bodyvtx_idx,
                                face_weight=1,
                                body_weight= 0
                                )
    objectives_pose.update({'lmk': lmk_err})
    print "\nstep 2: update head pose..."
    ch.minimize( fun      = objectives_pose,
                 x0       = [ g_scale,g_trans_2d,model.pose[3:6]],
                 method   = 'dogleg',
                 callback = on_step,
                 options  = opt_options )
    timer_end = time()
    print "step 2: head pose fitting done, in %f sec\n" % ( timer_end - timer_start )
    print "model.trans"
    print model.trans
    print "head.rot"
    print model.pose[3:6]
    print "all pose"
    print model.pose[:]
    print "g_scale"
    print g_scale
    print "g_trans_2d"
    print g_trans_2d
    output_dir = 'L:/yuanqing/imgs/imgs/vrn_result/niutou/01/'
    output_path = join(output_dir, 'step_2.obj')
    g_v =  g_scale.r*model.r[:,:]
    g_v[:,0:2] = g_v[:,0:2]+g_trans_2d.r
    write_simple_obj(mesh_v=g_v, mesh_f=model.f, filepath=output_path, verbose=False)
    # step 3: update all pose

    lmk_err = landmark_error_3d(mesh_verts=model,
                                scale=g_scale,
                                trans_2d=g_trans_2d,
                                target_lmk_3d_face=lmk_3d[0:68,:],
                                target_lmk_3d_body=lmk_3d[68:72,:],
                                lmk_facevtx_idx=lmk_facevtx_idx,
                                lmk_bodyvtx_idx=lmk_bodyvtx_idx,
                                face_weight=1,
                                body_weight= 1
                                )
    objectives_pose.update({'lmk': lmk_err})
    print "\nstep 3: update all pose..."
    ch.minimize( fun      = objectives_pose,
                 x0       = [ g_scale,g_trans_2d,model.pose[0:6]],
                 method   = 'dogleg',
                 callback = on_step,
                 options  = opt_options )
    timer_end = time()
    print "step 3: head pose fitting done, in %f sec\n" % ( timer_end - timer_start )
    print "model.trans"
    print model.trans
    print "head.rot"
    print model.pose[3:6]
    print "all pose"
    print model.pose[:]
    print "g_scale"
    print g_scale
    print "g_trans_2d"
    print g_trans_2d
    output_dir = 'L:/yuanqing/imgs/imgs/vrn_result/niutou/01/'
    output_path = join(output_dir, 'step_3.obj')
    g_v =  g_scale.r*model.r[:,:]
    g_v[:,0:2] = g_v[:,0:2]+g_trans_2d.r
    write_simple_obj(mesh_v=g_v, mesh_f=model.f, filepath=output_path, verbose=False)

    '''
    # step 2: non-rigid alignment
    timer_start = time()
    print "step 2: start non-rigid fitting..."
    ch.minimize(fun=objectives,
                x0=free_variables,
                method='dogleg',
                callback=on_step,
                options=opt_options)
    timer_end = time()
    print "step 2: fitting done, in %f sec\n" % (timer_end - timer_start)
    '''
    # return results
    parms = {'trans': model.trans.r, 'pose': model.pose.r, 'betas': model.betas.r}

    return g_v, model.f, parms  # ,align_v,align_f
    #return model.r, model.f, parms  # ,align_v,align_f