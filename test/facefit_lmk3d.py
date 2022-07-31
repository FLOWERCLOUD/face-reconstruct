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
from fitting.util import load_binary_pickle, write_simple_obj, safe_mkdir,mat_save
import scipy.io as sio
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import myplot.vtkplot as vp
import quaternion
# -----------------------------------------------------------------------------

def fit_lmk3d( lmk_3d,                      # input landmark 3d
               model,                       # model
               lmk_face_idx, lmk_b_coords,  # landmark embedding
               weights,                     # weights for the objectives
               shape_num=300, expr_num=100, opt_options=None ):
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
    shape_idx      = np.arange( 0, min(300,shape_num) )        # valid shape component range in "betas": 0-299
    expr_idx       = np.arange( 300, 300+min(100,expr_num) )   # valid expression component range in "betas": 300-399
    used_idx       = np.union1d( shape_idx, expr_idx )
    model.betas[:] = np.random.rand( model.betas.size ) * 0.0  # initialized to zero
    model.pose[:]  = np.random.rand( model.pose.size ) * 0.0   # initialized to zero
    #free_variables = [ model.trans, model.pose, model.betas[used_idx] ]
    free_variables = [model.pose, model.betas[used_idx]]
    # weights
    print "fit_lmk3d(): use the following weights:"
    for kk in weights.keys():
        print "fit_lmk3d(): weights['%s'] = %f" % ( kk, weights[kk] ) 

    # objectives
    # lmk
    scale = ch.array([1])
    lmk_err = landmark_error_3d( mesh_verts=scale*model,
                                 mesh_faces=model.f, 
                                 lmk_3d=lmk_3d, 
                                 lmk_face_idx=lmk_face_idx, 
                                 lmk_b_coords=lmk_b_coords, 
                                 weight=weights['lmk'] )
    # regularizer
    shape_err = weights['shape'] * model.betas[shape_idx] 
    expr_err  = weights['expr']  * model.betas[expr_idx] 
    pose_err  = weights['pose']  * model.pose[3:] # exclude global rotation
    objectives = {}
    objectives.update( { 'lmk': lmk_err, 'shape': shape_err, 'expr': expr_err, 'pose': pose_err } ) 

    # options
    if opt_options is None:
        print "fit_lmk3d(): no 'opt_options' provided, use default settings."
        import scipy.sparse as sp
        opt_options = {}
        opt_options['disp']    = 1
        opt_options['delta_0'] = 0.1
        opt_options['e_3']     = 1e-4
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
    # step 2: non-rigid alignment
    timer_start = time()
    print "step 2: start non-rigid fitting..."    
    ch.minimize( fun      = objectives,
                 x0       = free_variables,
                 method   = 'dogleg',
                 callback = on_step,
                 options  = opt_options )
    timer_end = time()
    print "step 2: fitting done, in %f sec\n" % ( timer_end - timer_start )

    # return results
    parms = { 'trans': model.trans.r, 'pose': model.pose.r, 'betas': model.betas.r }
    return scale*model.r, model.f, parms #,align_v,align_f

# -----------------------------------------------------------------------------

def run_fitting():

    # input landmarks
    lmk_path = '../data/landmark_3d.pkl'
    lmk_3d = load_binary_pickle( lmk_path )

    q1 = quaternion.from_rotation_vector([0,0,0])
 #   quaternion.from_euler_angles(20,0,0)
    g_r = quaternion.as_rotation_matrix(q1)
    lmk_3d = np.asmatrix((lmk_3d+[0,0.01,-0.01])) *np.asmatrix(g_r)
    lmk_3d = np.asarray(lmk_3d)
    print "loaded 3d landmark from:", lmk_path
    fig = plt.figure(figsize=plt.figaspect(1))
    ax = plt.subplot(111, projection='3d')
    ax.scatter(lmk_3d[:,0], lmk_3d[:,1], lmk_3d[:,2], c='r')
    ax.set_zlabel('Z')
    ax.set_ylabel('Y')
    ax.set_xlabel('X')
    ax.set_xlim(ax.get_xlim()[::-1])
#    plt.show()

    # model
    model_path = './models/male_model.pkl' # change to 'female_model.pkl' or 'generic_model.pkl', if needed
    model = load_model( model_path )       # the loaded model object is a 'chumpy' object, check https://github.com/mattloper/chumpy for details
    print "loaded model from:", model_path

    # landmark embedding
    lmk_emb_path = '../data/lmk_embedding_intraface_to_flame.pkl'
    lmk_face_idx, lmk_b_coords = load_embedding( lmk_emb_path )

    mat_save({'lmk_face_idx':lmk_face_idx}, '../lmk_face_idx.mat')
    mat_save({'lmk_b_coord':lmk_b_coords}, '../lmk_b_coords.mat')
    print "loaded lmk embedding"

    # output
    output_dir = '../output'
    safe_mkdir( output_dir )

    # weights
    weights = {}
    weights['lmk']   = 1.0
    weights['shape'] = 0.001
    weights['expr']  = 0.001
    weights['pose']  = 0.1
    
    # optimization options
    import scipy.sparse as sp
    opt_options = {}
    opt_options['disp']    = 1
    opt_options['delta_0'] = 0.1
    opt_options['e_3']     = 1e-4
    opt_options['maxiter'] = 100
    sparse_solver = lambda A, x: sp.linalg.cg(A, x, maxiter=opt_options['maxiter'])[0]
    opt_options['sparse_solver'] = sparse_solver

    # run fitting
    mesh_v, mesh_f, parms = fit_lmk3d( lmk_3d=lmk_3d,                                         # input landmark 3d
                                       model=model,                                           # model
                                       lmk_face_idx=lmk_face_idx, lmk_b_coords=lmk_b_coords,  # landmark embedding
                                       weights=weights,                                       # weights for the objectives
                                       shape_num=300, expr_num=100, opt_options=opt_options ) # options
#    vp.trisurf(align_v, align_f, rendertype='wireframe')
#    vp.scatter(pts=lmk_3d,alpha=1,mode='sphere',scale=0.001)
    # vp.trisurf(v_final, f_final, rendertype='wireframe', color3f=(0,0,0), alpha=0.3)
#    vp.show()
    # write result
    output_path = join( output_dir, 'fit_lmk3d_result.obj' )


    vp.trisurf(mesh_v, mesh_f, rendertype='wireframe')
    vp.scatter(pts=lmk_3d,alpha=1,mode='sphere',scale=0.001)
    # vp.trisurf(v_final, f_final, rendertype='wireframe', color3f=(0,0,0), alpha=0.3)
    vp.show()
    write_simple_obj( mesh_v=mesh_v, mesh_f=mesh_f, filepath=output_path, verbose=False )

# -----------------------------------------------------------------------------

if __name__ == '__main__':

    run_fitting()

