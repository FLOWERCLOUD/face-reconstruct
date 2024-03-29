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
from configs.config import igl_python_path
sys.path.insert(0, igl_python_path)
import pyigl as igl
import scipy.sparse as sp

opt_options_10 = {}
opt_options_10['disp'] = 1
opt_options_10['delta_0'] = 0.1
opt_options_10['e_3'] = 1e-4
opt_options_10['maxiter'] = 10
sparse_solver_10 = lambda A, x: sp.linalg.cg(A, x, maxiter=opt_options_10['maxiter'])[0]
opt_options_10['sparse_solver'] = sparse_solver_10

opt_options_20 = {}
opt_options_20['disp'] = 1
opt_options_20['delta_0'] = 0.1
opt_options_20['e_3'] = 1e-4
opt_options_20['maxiter'] = 20
sparse_solver_20 = lambda A, x: sp.linalg.cg(A, x, maxiter=opt_options_20['maxiter'])[0]
opt_options_20['sparse_solver'] = sparse_solver_20

opt_options_50 = {}
opt_options_50['disp'] = 1
opt_options_50['delta_0'] = 0.1
opt_options_50['e_3'] = 1e-4
opt_options_50['maxiter'] = 50
sparse_solver_50 = lambda A, x: sp.linalg.cg(A, x, maxiter=opt_options_50['maxiter'])[0]
opt_options_50['sparse_solver'] = sparse_solver_50



def fit_lmk3d(lmk_3d,  # input landmark 3d
              model,  # model
              mesh_faces,
              lmk_face_idx, lmk_b_coords,
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


    g_scale = ch.array([1000])
    g_trans_2d = ch.array([200,200,0])


    print "\nstep 1: 根据face landmark 来获得3维位移和放缩..."
    timer_start = time()
    face_lmk_err,body_lmk_err = landmark_error_3d(
                                scale = g_scale,
                                trans_2_3d = g_trans_2d,
                                mesh_verts=model,
                                mesh_faces=mesh_faces,
                                target_lmk_3d_face=lmk_3d[0:68,:],
                                target_lmk_3d_body=lmk_3d[68:72,0:2],
                                lmk_face_idx=lmk_face_idx,
                                lmk_b_coords=lmk_b_coords,
                                lmk_facevtx_idx=lmk_facevtx_idx,
                                lmk_bodyvtx_idx=lmk_bodyvtx_idx,
                                face_weight=1,
                                body_weight= 0
                                )
    objectives_pose = {}
    objectives_pose.update({'face_lmk': face_lmk_err,'body_lmk':body_lmk_err})

    ch.minimize( fun      = objectives_pose,
                 x0       = [ g_scale,g_trans_2d],
                 method   = 'dogleg',
                 callback = on_step,
                 options  = opt_options_50 )
    timer_end = time()

    output_dir = 'L:/yuanqing/imgs/imgs/vrn_result/niutou/01/'
    def print_para_resut(step,timer_end,timer_start):
        print step
        print "in %f sec\n" % (timer_end - timer_start)
        print "model.trans"
        print model.trans
        print "all pose"
        print model.pose[:]
        print "g_scale"
        print g_scale
        print "g_trans_2d"
        print g_trans_2d

    def debug_result(output_path,filename):
        output_path = output_path+filename
        g_v = g_scale.r * model.r[:, :]
        g_v[:, :] = g_v[:, :] + g_trans_2d.r
        write_simple_obj(mesh_v=g_v, mesh_f=model.f, filepath=output_path, verbose=False)

    print_para_resut('step1',timer_end,timer_start)
    debug_result(output_dir,'step_1.obj')

    print "\nstep 2: 根据body landmark 来获得2维位移和旋转..."
    timer_start = time()
    face_lmk_err, body_lmk_err = landmark_error_3d(
                                scale = g_scale,
                                trans_2_3d=g_trans_2d,
                                mesh_verts=model,
                                mesh_faces=mesh_faces,
                                target_lmk_3d_face=lmk_3d[0:68,:],
                                target_lmk_3d_body=lmk_3d[68:72,0:2],
                                lmk_face_idx=lmk_face_idx,
                                lmk_b_coords=lmk_b_coords,
                                lmk_facevtx_idx=lmk_facevtx_idx,
                                lmk_bodyvtx_idx=lmk_bodyvtx_idx,
                                face_weight=1,
                                body_weight= 0
                                )
    objectives_pose.update({'face_lmk': face_lmk_err,'body_lmk':body_lmk_err})
    ch.minimize( fun      = objectives_pose,
                 x0       = [ g_trans_2d,model.pose[0:3]],
                 method   = 'dogleg',
                 callback = on_step,
                 options  = opt_options_10 )
    face_lmk_err, body_lmk_err = landmark_error_3d(
                                scale = g_scale,
                                trans_2_3d=g_trans_2d,
                                mesh_verts=model,
                                mesh_faces=mesh_faces,
                                target_lmk_3d_face=lmk_3d[0:68,:],
                                target_lmk_3d_body=lmk_3d[68:72,0:2],
                                lmk_face_idx=lmk_face_idx,
                                lmk_b_coords=lmk_b_coords,
                                lmk_facevtx_idx=lmk_facevtx_idx,
                                lmk_bodyvtx_idx=lmk_bodyvtx_idx,
                                face_weight=1,
                                body_weight= 0
                                )
    objectives_pose.update({'face_lmk': face_lmk_err,'body_lmk':body_lmk_err})
    ch.minimize( fun      = objectives_pose,
                 x0       = [ g_trans_2d,model.pose[0:3]],
                 method   = 'dogleg',
                 callback = on_step,
                 options  = opt_options_10 )
    timer_end = time()
    print_para_resut('step2',timer_end,timer_start)
    debug_result(output_dir,'step_2.obj')


    print "\nstep 3: update g_scale,g_trans_2d,model.pose[0:6] ..."
    timer_start = time()
    face_lmk_err, body_lmk_err = landmark_error_3d(
                                scale=g_scale,
                                trans_2_3d=g_trans_2d,
                                mesh_verts=model,
                                mesh_faces=mesh_faces,
                                target_lmk_3d_face=lmk_3d[0:68,:],
                                target_lmk_3d_body=lmk_3d[68:72,0:2],
                                lmk_face_idx=lmk_face_idx,
                                lmk_b_coords=lmk_b_coords,
                                lmk_facevtx_idx=lmk_facevtx_idx,
                                lmk_bodyvtx_idx=lmk_bodyvtx_idx,
                                face_weight=1,
                                body_weight= 0.0
                                )
    objectives_pose.update({'face_lmk': face_lmk_err, 'body_lmk': body_lmk_err})
    ch.minimize( fun      = objectives_pose,
                 x0       = [ g_scale,g_trans_2d],
                 method   = 'dogleg',
                 callback = on_step,
                 options  = opt_options_20 )
    timer_end = time()
    print_para_resut('step3',timer_end,timer_start)
    debug_result(output_dir,'step_3.obj')

    print "\nstep 4: update model.pose[6:9],model.beta ..."
    # regularizer
    shape_err = weights['shape'] * model.betas[shape_idx]
    expr_err = weights['expr'] * model.betas[expr_idx]
    pose_err = weights['pose'] * model.pose[3:]  # exclude global rotation
    #    objectives = {}
    #    objectives.update({'lmk': lmk_err, 'shape': shape_err, 'expr': expr_err, 'pose': pose_err})


    timer_start = time()
    face_lmk_err, body_lmk_err = landmark_error_3d(
                                scale=g_scale,
                                trans_2_3d=g_trans_2d,
                                mesh_verts=model,
                                mesh_faces=mesh_faces,
                                target_lmk_3d_face=lmk_3d[0:68,:],
                                target_lmk_3d_body=lmk_3d[68:72,0:2],
                                lmk_face_idx = lmk_face_idx,
                                lmk_b_coords= lmk_b_coords,
                                lmk_facevtx_idx=lmk_facevtx_idx,
                                lmk_bodyvtx_idx=lmk_bodyvtx_idx,
                                face_weight=0.001,
                                body_weight= 0.0
                                )
    objectives_pose.update({'face_lmk': face_lmk_err,'body_lmk':body_lmk_err, 'shape': shape_err, 'expr': expr_err, 'pose': pose_err})
    ch.minimize( fun      = objectives_pose,
                 x0       = [model.pose,model.betas[0:400]],
                 method   = 'dogleg',
                 callback = on_step,
                 options  = opt_options_20 )
    timer_end = time()
    print_para_resut('step4',timer_end,timer_start)
    debug_result(output_dir,'step_4.obj')


    # return results
    parms = {'trans': model.trans.r, 'pose': model.pose.r, 'betas': model.betas.r}
    g_v = g_scale.r * model.r[:, :]
    g_v[:, :] = g_v[:, :] + g_trans_2d.r
    return g_v, model.f, parms  # ,align_v,align_f
    #return model.r, model.f, parms  # ,align_v,align_f