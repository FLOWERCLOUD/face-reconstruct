# -- coding: utf-8 --
import os
import numpy as np
import cPickle as pickle
import scipy.io as scio
import sys
sys.path.insert(0, "E:/workspace/igl_python/")
import pyigl as igl
import math
from fitting.util import safe_mkdirs, mat_save,mat_load,IglMatrixTonpArray,FileFilt,write_simple_obj,k_main_dir_sklearn
from fitting.util import readImage,write_image_and_featurepoint,read_landmark, \
    cast2d_to3d_trimesh,scaleToOriCoodi_bottomleft,sym_plane,corr_point,\
    sym_point,write_landmark_to_obj,write_full_obj,corr_landmark_tofit_data,detect_68_landmark_dlib,get_vertex_normal,read_igl_obj,save_binary_pickle,load_binary_pickle,readVertexColor,\
    get_vertex_normal,sample_color_from_img

# triangluration of 68 landmark
landmark_face = np.array([[57,8,9],[57,9,56],[56,9,10],[56,10,55],[55,10,11],[55,11,54],[54,11,12],
[54,12,13],[54,13,35],[13,14,35],[14,47,35],[14,46,47],[14,15,46],[15,45,46],
[15,16,45],[16,26,45],[26,25,45],[25,44,45],[25,24,44],[24,43,44],[24,23,43],
[23,22,43],[22,42,43],[22,27,42],[27,28,42],[28,29,42],[42,29,47],[47,29,35],
[29,30,35],[30,34,35],[30,33,34],[33,51,52],[33,52,34],[34,52,35],[35,52,53],
[35,53,54],[51,62,52],[52,62,63],[52,63,53],[53,63,64],[53,64,54],[64,55,54],
[64,65,55],[65,56,55],[65,66,56],[66,57,56],
[57,7,8],[57,58,7],[58,6,7],[58,59,6],[59,5,6],[59,48,5],[48,4,5],[48,3,4],
[31,3,48],[31,2,3],[31,40,2],[40,41,2],[41,1,2],[36,1,41],[36,0,1],[36,17,0],
[36,18,17],[36,37,18],[37,19,18],[37,38,19],[19,38,20],[20,38,21],[21,38,39],
[21,39,27],[27,39,28],[28,39,29],[39,40,29],[40,31,29],[29,31,30],[30,31,32],
[30,32,33],[31,48,49],[31,49,50],[31,50,32],[32,50,33],[33,50,51],[48,60,49],
[49,60,61],[49,61,50],[50,61,62],[50,62,51],[48,59,60],[59,67,60],[67,59,58],
[67,58,66],[66,58,57]])
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

def generate_3d_landmark_from_vrn_model(oriimage_file_path,mat_file_path,vrn_landmark_path,output_dir):
    safe_mkdirs(output_dir)
    result = mat_load(mat_file_path)
    BB = result['BB'][0]
    result = result['result']
    V= result['vertices'][0,0]
    V[:, 1] = 192 -1- V[:, 1]  # 以左下角为原点
    F = result['faces'][0, 0]
    F = F - 1 # 以0 为坐标的序号
    vtx_color = result['FaceVertexCData'][0, 0] #255
    ori_image = readImage(oriimage_file_path)
    height, width, dim = ori_image.shape
    use_dlib = 0
    if use_dlib:
        landmark_2d = detect_68_landmark_dlib(ori_image)
        write_image_and_featurepoint(ori_image, landmark_2d, output_dir + '/' + 'dlib_feature_2d' + '.jpg')
        landmark_2d[:, 1] = height-1 - landmark_2d[:, 1]

    else:
        landmark_2d = read_landmark(vrn_landmark_path)
        landmark_2d[:, 1] = height - 1 - landmark_2d[:, 1]  # 坐标系原点转化为左下角
    BB[1] = height - 1 - BB[1] - BB[2]  # 坐标系原点转化为左下角
    target_V = scaleToOriCoodi_bottomleft(V, BB, 192)

    target_Vn =   get_vertex_normal(target_V,F)
    vrn_scale_to_ori_path = output_dir+'/'+'vrn_scale_to_orig.obj'
    write_full_obj(target_V,F,target_Vn,F,np.array([]),np.array([]),vtx_color,vrn_scale_to_ori_path)
# 3 .通过二维投影的方式求三维点
    from time import time
    timer_start = time()
    index_triangles, index_ray, result_locations = cast2d_to3d_trimesh(vrn_scale_to_ori_path,landmark_2d)
    bool_ray = np.zeros((landmark_2d.shape[0],1),np.int)
    for i_ray in index_ray:
        bool_ray[i_ray] = 1
    landmark_v = np.zeros((landmark_2d.shape[0],3),np.float64)
    for i in range(0,landmark_v.shape[0]):
        landmark_v[i,:] = [landmark_2d[i,0],landmark_2d[i,1],100]
    for i in range(0,result_locations.shape[0]):
        landmark_idx = index_ray[i]
        landmark_v[landmark_idx,:] = result_locations[i,:]
    igl.writeOBJ(output_dir + '/cast_landmark_step1' + '.obj', igl.eigen.MatrixXd(landmark_v.astype('float64')),
                igl.eigen.MatrixXi(landmark_face.astype('intc')))
    return landmark_v
'''
vrn_object_dir : vrn 某个人物文件所在目录
object_name : 该人物的名字
out_put_dir : 生成文件的目录
'''
def generate_face( vrn_object_dir,object_name,out_put_dir,project_dir,
                   frame_model_path = './models/female_model.pkl',frame_lmk_emb_path='./data/lmk_embedding_intraface_to_flame.pkl',use_3d_landmark=False,pre_result = None  ):
    # if os.path.exists(out_put_dir + '/face_result.pkl'):
    #     result = load_binary_pickle( filepath=out_put_dir + '/face_result.pkl')
    #     return  result

    from smpl_webuser.serialization import load_model
    from fitting.landmarks import load_embedding, landmark_error_3d
    model = load_model(frame_model_path)  # the loaded model object is a 'chumpy' object, check https://github.com/mattloper/chumpy for details
    # landmark embedding
    lmk_face_idx, lmk_b_coords = load_embedding(frame_lmk_emb_path)
    #68 个 landmark
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
    safe_mkdirs(out_put_dir)
    if not use_3d_landmark:
        if 0:
            generate_3d_landmark_from_vrn_model(
                oriimage_file_path =vrn_object_dir+'/'+object_name+'.jpg',
                mat_file_path =vrn_object_dir+'/'+'result/'+object_name+'.mat',
                #vrn_landmark_path=vrn_object_dir+'/'+'2d/'+object_name+'.txt',
                vrn_landmark_path='E:\workspace\dataset\zhengjianzhao_seg_dataset/bgBlue\Landmark/'  + object_name + '.txt',
                output_dir=out_put_dir)

            landmark_v, landmark_f, t, t_f, n, n_f =read_igl_obj(out_put_dir+'/cast_landmark_step1.obj')
        else:
            if os.path.exists(project_dir+'Landmark/'+object_name + '.txt'):
                pass
                hm_3d_path =project_dir+'Landmark/'+object_name + '.txt'
            else:
                hm_3d_path = vrn_object_dir + '/2d/' + object_name + '.txt'
                if not os.path.exists(hm_3d_path):
                    hm_3d_path = vrn_object_dir + '/3d/' + object_name + '.txt'
                    if not os.path.exists(hm_3d_path):
                        return
            oriimage_file_path = vrn_object_dir + '/' + object_name + '.jpg'
            if not os.path.exists(oriimage_file_path):
                oriimage_file_path = vrn_object_dir + '/' + object_name + '.jpeg'
                if not os.path.exists(oriimage_file_path):
                    oriimage_file_path = vrn_object_dir + '/' + object_name + '.png'
                else:
                    return
            ori_image = readImage(oriimage_file_path)
            print oriimage_file_path
            height, width, dim = ori_image.shape
            landmark_2d = read_landmark(hm_3d_path)  # 2d landmark 是以1为起点的
            landmark_2d =landmark_2d-1
            landmark_2d[:, 1] = height - 1 - landmark_2d[:, 1]  # 坐标系原点转化为左下角
            landmark_v = landmark_2d
            tmp = np.zeros((landmark_2d.shape[0],1))
            landmark_v =np.hstack((landmark_2d,tmp))
            #write_landmark_to_obj(out_put_dir + '/landmark3d.obj', landmark_v)
    else:
        hm_3d_path = vrn_object_dir+'/3d/'+object_name+'_hm.txt'
        oriimage_file_path = vrn_object_dir +'/3d/' + 'figure_3d'+ '.jpg'
        ori_image = readImage(oriimage_file_path)
        height, width, dim = ori_image.shape
        landmark_2d = read_landmark(hm_3d_path) #3d landmark 是以0为起点的
        landmark_2d[:, 1] = height - 1 - landmark_2d[:, 1]  # 坐标系原点转化为左下角
        landmark_v = landmark_2d
        write_landmark_to_obj(out_put_dir+'/landmark3d.obj', landmark_v)
        pass


    landmark_body = np.array([[586,336,130],[562,369,150],[709,295,160],[727,262,150]])
    landmark_body[:,1] = 683-landmark_body[:,1]
    target_lmk_v = np.concatenate((landmark_v[:, 0:3], landmark_body[:, 0:3]))  # 三维landmark点

    # weights
    weights = {}
    weights['lmk'] = 1.0
    weights['shape'] = 0.001
    weights['expr'] = 0.001
    weights['pose'] = 0.1
    # default optimization options
    import scipy.sparse as sp
    opt_options = {}
    opt_options['disp'] = 1
    opt_options['delta_0'] = 0.1
    opt_options['e_3'] = 1e-4
    opt_options['maxiter'] = 10
    sparse_solver = lambda A, x: sp.linalg.cg(A, x, maxiter=opt_options['maxiter'])[0]
    opt_options['sparse_solver'] = sparse_solver



    mesh_v, mesh_f, parms = fit_lmk3d(lmk_3d=target_lmk_v,  # input landmark 3d
                                      model=model,  # model
                                      mesh_faces=model.f,
                                      lmk_facevtx_idx=face_select_lmk, lmk_bodyvtx_idx=body_select_lmk,  # landmark embedding
                                      lmk_face_idx=lmk_face_idx,
                                      lmk_b_coords=lmk_b_coords,
                                      weights=weights,  # weights for the objectives
                                      shape_num=300, expr_num=100, opt_options=opt_options,
                                      output_dir=out_put_dir,use_3d_landmark=use_3d_landmark,pre_result=pre_result)  # options
    face_result ={'mesh_v':mesh_v,'mesh_f':mesh_f,'parms':parms}
    # output_path = out_put_dir+ '/face.obj'
    # write_simple_obj(mesh_v=mesh_v, mesh_f=mesh_f, filepath=output_path, verbose=False)
    vertex_normal = get_vertex_normal(mesh_v,mesh_f)
    ori_image = ori_image[::-1,:,:]
    v_color = sample_color_from_img(mesh_v, vertex_normal, ori_image)
    frame_re_texture_map = 'D:\mproject/face-reconstruct/texpc\source_para/texture/frame_template_retex.obj'
    v_frame_aligned, f_frame_aligned, t_frame_aligned, t_f_frame_aligned, n_frame_aligned, n_f_frame_aligned = read_igl_obj(
        frame_re_texture_map)
    write_full_obj(mesh_v, mesh_f, vertex_normal, n_f_frame_aligned, t_frame_aligned, t_f_frame_aligned, v_color,out_put_dir+'/face_with_texture.obj')

    subdived_mesh, subdived_mesh_f, t_frame_aligned, t_f_frame_aligned, n_frame_aligned, n_f_frame_aligned = read_igl_obj(
        out_put_dir+'/face_with_texture.obj')
    vtx_color = v_color

    from fitting.util import mesh_loop
    mesh_loop(subdived_mesh, subdived_mesh_f, n_frame_aligned, n_f_frame_aligned, t_frame_aligned, t_f_frame_aligned,
              vtx_color,
              out_put_dir+'face_with_test_subdiv.obj', 2)
    subdived_mesh, subdived_mesh_f, t_frame_aligned, t_f_frame_aligned, n_frame_aligned, n_f_frame_aligned = read_igl_obj(
        out_put_dir+'face_with_test_subdiv.obj')
    v_color = sample_color_from_img(subdived_mesh, n_frame_aligned, ori_image)
    write_full_obj(subdived_mesh, subdived_mesh_f, n_frame_aligned, n_f_frame_aligned, t_frame_aligned, t_f_frame_aligned, v_color,out_put_dir+'/subface_with_texture.obj')
    save_binary_pickle(face_result, out_put_dir+'/face_result.pkl')
    return  face_result

def fit_lmk3d(lmk_3d,  # input landmark 3d
              model,  # model
              mesh_faces,
              lmk_facevtx_idx, lmk_bodyvtx_idx,  # landmark embedding
              lmk_face_idx,
              lmk_b_coords,
              weights,  # weights for the objectives
              shape_num=300, expr_num=100, opt_options=None,output_dir=None,use_3d_landmark =False,pre_result =None):
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
    from fitting.landmarks import landmark_error_3d
    import chumpy as ch



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

    g_scale = ch.array([180])
    g_trans_2d = ch.array([200, 300, 0])

    if pre_result != None:
        result = pre_result
        v = result['mesh_v']
        face = result['mesh_f']
        parms = result['parms']
        T = parms['trans']
        pose = parms['pose']
        betas = parms['betas']
        R = parms['global_rotate']
        Scale = parms['scale']
        model.betas[:] =  betas[:]



    print "\nstep 1: 根据face landmark 来获得2维位移和放缩..."
    timer_start = time()
    face_lmk_err, body_lmk_err = landmark_error_3d(
        scale=g_scale,
        trans_2_3d=g_trans_2d,
        mesh_verts=model,
        mesh_faces=mesh_faces,
        target_lmk_3d_face=lmk_3d[0:68, 0:3],
        target_lmk_3d_body=lmk_3d[68:72, :],
        lmk_face_idx=lmk_face_idx,
        lmk_b_coords=lmk_b_coords,
        lmk_facevtx_idx=lmk_facevtx_idx,
        lmk_bodyvtx_idx=lmk_bodyvtx_idx,
        face_weight=1,
        body_weight=0,
        use_lunkuo=True,
        use_3d_landmark=use_3d_landmark
    )
    objectives_pose = {}
    objectives_pose.update({'face_lmk': face_lmk_err, 'body_lmk': body_lmk_err})

    ch.minimize(fun=objectives_pose,
                x0=[g_scale, g_trans_2d],
                method='dogleg',
                callback=on_step,
                options=opt_options_50)
    timer_end = time()

    #    output_dir = 'L:/yuanqing/imgs/imgs/vrn_result/niutou/10/'
    def print_para_resut(step, timer_end, timer_start):
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

    from  para_texture_buid import add_texture_and_write
    def debug_result(output_path, filename):
        return
        output_path = output_path + filename
        g_v = g_scale.r * model.r[:, :]
        g_v[:, :] = g_v[:, :] + g_trans_2d.r
        # write_simple_obj(mesh_v=g_v, mesh_f=model.f, filepath=output_path, verbose=False)
        add_texture_and_write(v_np=g_v, f_np=model.f, output_path=output_path)

    print_para_resut('step1', timer_end, timer_start)
    debug_result(output_dir, 'step_2d_1.obj')

    print "\nstep 2: 根据body landmark 来获得2维位移和旋转..."
    timer_start = time()
    face_lmk_err, body_lmk_err = landmark_error_3d(
        scale=g_scale,
        trans_2_3d=g_trans_2d,
        mesh_verts=model,
        mesh_faces=mesh_faces,
        target_lmk_3d_face=lmk_3d[0:68,  0:3],
        target_lmk_3d_body=lmk_3d[68:72, :],
        lmk_face_idx=lmk_face_idx,
        lmk_b_coords=lmk_b_coords,
        lmk_facevtx_idx=lmk_facevtx_idx,
        lmk_bodyvtx_idx=lmk_bodyvtx_idx,
        face_weight=1,
        body_weight=0,
        use_lunkuo=True,
        use_3d_landmark=use_3d_landmark
    )
    objectives_pose.update({'face_lmk': face_lmk_err, 'body_lmk': body_lmk_err})
    ch.minimize(fun=objectives_pose,
                x0=[g_trans_2d, model.pose[0:3]],
                method='dogleg',
                callback=on_step,
                options=opt_options_10)
    face_lmk_err, body_lmk_err = landmark_error_3d(
        scale=g_scale,
        trans_2_3d=g_trans_2d,
        mesh_verts=model,
        mesh_faces=mesh_faces,
        target_lmk_3d_face=lmk_3d[0:68,  0:3],
        target_lmk_3d_body=lmk_3d[68:72, :],
        lmk_face_idx=lmk_face_idx,
        lmk_b_coords=lmk_b_coords,
        lmk_facevtx_idx=lmk_facevtx_idx,
        lmk_bodyvtx_idx=lmk_bodyvtx_idx,
        face_weight=1,
        body_weight=0,
        use_lunkuo=True,
        use_3d_landmark=use_3d_landmark
    )
    objectives_pose.update({'face_lmk': face_lmk_err, 'body_lmk': body_lmk_err})
    ch.minimize(fun=objectives_pose,
                x0=[g_trans_2d, model.pose[0:3]],
                method='dogleg',
                callback=on_step,
                options=opt_options_10)
    timer_end = time()
    print_para_resut('step2', timer_end, timer_start)
    debug_result(output_dir, 'step_2d_2.obj')

    print "\nstep 3: update g_scale,g_trans_2d,model.pose[0:6] ..."
    timer_start = time()
    face_lmk_err, body_lmk_err = landmark_error_3d(
        scale=g_scale,
        trans_2_3d=g_trans_2d,
        mesh_verts=model,
        mesh_faces=mesh_faces,
        target_lmk_3d_face=lmk_3d[0:68,  0:3],
        target_lmk_3d_body=lmk_3d[68:72, :],
        lmk_face_idx=lmk_face_idx,
        lmk_b_coords=lmk_b_coords,
        lmk_facevtx_idx=lmk_facevtx_idx,
        lmk_bodyvtx_idx=lmk_bodyvtx_idx,
        face_weight=1,
        body_weight=0,
        use_lunkuo=False,
        use_3d_landmark=use_3d_landmark
    )
    objectives_pose.update({'face_lmk': face_lmk_err, 'body_lmk': body_lmk_err})
    ch.minimize(fun=objectives_pose,
                x0=[g_scale, g_trans_2d, model.pose[0:3]],
                method='dogleg',
                callback=on_step,
                options=opt_options_20)
    timer_end = time()
    print_para_resut('step3', timer_end, timer_start)
    debug_result(output_dir, 'step_2d_3.obj')

    print "\nstep 4: update model.pose[6:9],model.beta ...,只使用landmark"
    # regularizer
    shape_err = weights['shape'] * model.betas[shape_idx]

    shape_weight = np.zeros(shape_idx.size)
    shape_weight[:] = weights['shape']
    range1 = range(0,10)
    range2 = range(10,50)
    shape_weight[range1] = weights['shape']*5
    shape_weight[range2] = weights['shape'] * 2
    # shape_weight[10] = weights['shape']
    # shape_weight[28] = weights['shape']
    #expr_err = weights['expr'] * model.betas[expr_idx]
    shape_err = shape_weight *model.betas[shape_idx]


    expr_err = weights['expr'] * model.betas[expr_idx]
    pose_err = weights['pose'] * model.pose[6:9]  # exclude global rotation
    #    objectives = {}
    #    objectives.update({'lmk': lmk_err, 'shape': shape_err, 'expr': expr_err, 'pose': pose_err})

    timer_start = time()
    face_lmk_err, body_lmk_err = landmark_error_3d(
        scale=g_scale,
        trans_2_3d=g_trans_2d,
        mesh_verts=model,
        mesh_faces=mesh_faces,
        target_lmk_3d_face=lmk_3d[0:68, 0:3],
        target_lmk_3d_body=lmk_3d[68:72, 0:2],
        lmk_face_idx=lmk_face_idx,
        lmk_b_coords=lmk_b_coords,
        lmk_facevtx_idx=lmk_facevtx_idx,
        lmk_bodyvtx_idx=lmk_bodyvtx_idx,
        face_weight=1 / g_scale.r * 0.8,
        body_weight=0.0,
        use_lunkuo=True, #使用轮廓的2d 投影
        use_3d_landmark=use_3d_landmark
    )

    objectives_pose.update(
        {'face_lmk': face_lmk_err, 'body_lmk': body_lmk_err, 'shape': shape_err, 'expr': expr_err,
         'pose': pose_err})
    if pre_result != None:
        pass
    else:
        ch.minimize(fun=objectives_pose,
                    #x0=[g_scale, model.betas[11:100], g_trans_2d,model.betas[302:350]], #, model.pose[3:6]
                    x0=[g_scale, model.betas[0:300], g_trans_2d,model.pose[0:3]],
                    method='dogleg',
                    callback=on_step,
                    options=opt_options_20)
    timer_end = time()
    print_para_resut('step4', timer_end, timer_start)
    debug_result(output_dir, 'step4_2d_4.obj')

    print "\nstep 5: update model.pose[6:9],model.beta ...,只使用landmark"
    # regularizer
    shape_err = weights['shape'] * model.betas[shape_idx]
    expr_err = weights['expr'] * model.betas[expr_idx]
    pose_err = weights['pose'] * model.pose[6:15]  # exclude global rotation
    #    objectives = {}
    #    objectives.update({'lmk': lmk_err, 'shape': shape_err, 'expr': expr_err, 'pose': pose_err})

    timer_start = time()
    face_lmk_err, body_lmk_err = landmark_error_3d(
        scale=g_scale,
        trans_2_3d=g_trans_2d,
        mesh_verts=model,
        mesh_faces=mesh_faces,
        target_lmk_3d_face=lmk_3d[0:68,  0:3],
        target_lmk_3d_body=lmk_3d[68:72, 0:2],
        lmk_face_idx=lmk_face_idx,
        lmk_b_coords=lmk_b_coords,
        lmk_facevtx_idx=lmk_facevtx_idx,
        lmk_bodyvtx_idx=lmk_bodyvtx_idx,
        face_weight=1 / g_scale.r * 0.8,
        body_weight=0.0,
        use_lunkuo=True, #使用轮廓的2d 投影
        use_3d_landmark=use_3d_landmark
        #use_landmark_idx = range(36,42) #只使用部分的landmark来求解表情
    )
    shape_err = weights['shape'] * (model.betas[shape_idx] - model.betas[shape_idx].r)
    express_weight = np.zeros(expr_idx.size)
    express_weight[:] = weights['expr']
    express_weight[0] = express_weight[0]*10
    express_weight[1] = express_weight[1] * 10
    #expr_err = weights['expr'] * model.betas[expr_idx]
    expr_err = express_weight *model.betas[expr_idx]
    pose_err = weights['pose'] * model.pose[6:15]  # exclude global rotation
    objectives_pose.update(
        {'face_lmk': face_lmk_err, 'body_lmk': body_lmk_err, 'shape': shape_err, 'expr': expr_err,
         'pose': pose_err})
    if pre_result != None:
        ch.minimize(fun=objectives_pose,
                    #x0=[model.betas[300:400],model.pose[6:9]],
                    x0=[g_trans_2d,model.pose[0:3],model.pose[6],model.betas[50:300]],
                    method='dogleg',
                    callback=on_step,
                    options=opt_options_20)
    else:
        pass
    timer_end = time()
    print_para_resut('step5', timer_end, timer_start)
    debug_result(output_dir, 'step5_2d_4.obj')



    global_rotate_pose = model.pose[0:3]
    print  global_rotate_pose
    import cv2
    global_rotate_pose = np.array([global_rotate_pose[0].r, global_rotate_pose[1].r, global_rotate_pose[2].r])
    print  global_rotate_pose
    global_rotate = cv2.Rodrigues(global_rotate_pose)[0]
    print global_rotate
    parms = {'trans': g_trans_2d.r, 'pose': model.pose.r, 'betas': model.betas.r,'scale':g_scale.r,'global_rotate':global_rotate}
    print parms
    g_v = g_scale.r * model.r[:, :]
    g_v[:, :] = g_v[:, :] + g_trans_2d.r
    return g_v, model.f, parms

g_count = 0
def shape_from_shading(input_model_dir,ori_img,seg_img,output_dir):
    import chumpy as ch
    from math import pi,cos
    if os.path.exists(input_model_dir + '/face_result.pkl'):
        result = load_binary_pickle( filepath=input_model_dir + '/face_result.pkl')
    # shape from shading
    from fitting.util import build_adjacency,get_vertex_normal
    ori_img[:,:] = ori_img[::-1,:]
    seg_img[:,:] = seg_img[::-1,:] # 转化为左下角

    v = result['mesh_v']
    face = result['mesh_f']
    parms = result['parms']
    T = parms['trans']
    pose = parms['pose']
    betas = parms['betas']
    R = parms['global_rotate']
    Scale = parms['scale']

    frame_model_path = './models/female_model.pkl'
    from smpl_webuser.serialization import load_model
    model = load_model(frame_model_path)
    model.pose[:] = pose[:]
    model.betas[:] = betas[:]
    Scale = ch.array(Scale)
    T= ch.array(T)
    model_v =Scale * model[:, :] + T
    adjacency = build_adjacency( model_v.r , model.f)
    vtx_normal = get_vertex_normal(model_v.r,model.f)
    select_vtx_idx =[]
    v_color = np.zeros((model_v.shape[0],3),np.uint8)
    for i in range(0,model_v.r.shape[0]):
        if i > 3930:
            break
        if np.dot(vtx_normal[i],np.array([0,0,1])) < cos(pi/3):
            continue
        if model_v.r[i,2]<20:
            continue
        vtx = model_v.r[i,:]
        x = int(vtx[0])
        y = int(vtx[1])
        if seg_img[y,x] :
            select_vtx_idx.append(i)
            v_color[i,:] = [255,0,0]
    print len(select_vtx_idx)
    write_full_obj(model_v.r, model.f, np.array([]), np.array([]), np.array([]), np.array([]), v_color,
                   output_dir+'/debug_model.obj')
    para_input ='D:/mproject/face-reconstruct/texpc/target_para/frame_tex.pkl'
    data = load_binary_pickle(para_input)
    tx_pc =data['texPC']
    mean_colors = data['texMU']
    texEV = data['texEV']
    select_mean_color = mean_colors[select_vtx_idx]
    select_tx_pc = tx_pc[:,select_vtx_idx,:]
    coeff = ch.zeros(199)
    coeff[0] =0.1
    color_sum = ch.zeros((len(select_vtx_idx),3))
    color_sum+=select_mean_color
    for p in range(0, 199):
        color_sum+=coeff[p]*select_tx_pc[p,:,:]

    img_vtx_color = ch.zeros((len(select_vtx_idx),3))
    for i in range(0,len(select_vtx_idx)):
        vtx_idx = select_vtx_idx[i]
        vtx = model_v.r[vtx_idx, :]
        x = int(vtx[0])
        y = int(vtx[1])
        b = ori_img[y, x, 0]
        g = ori_img[y, x, 1]
        r = ori_img[y, x, 2]
        img_vtx_color[i,:] =[r,g,b]
    color_term = color_sum - img_vtx_color
    objectives_pose ={}
    objectives_pose.update(
        {'data term': color_term,  'reg': (1/texEV)*coeff})

    def on_step1(_):
        print 'call back'
    para_color = np.zeros((model_v.r.shape[0],3))
    para_color[select_vtx_idx] = color_sum
    write_full_obj(model_v.r, model.f, np.array([]), np.array([]), np.array([]), np.array([]), para_color,
                   output_dir+'/debug_para_color_pre.obj')
    ch.minimize(fun=objectives_pose,
                x0=[coeff[0:3]], #, model.pose[3:6]
                method='dogleg',
                callback=on_step1,
                options=opt_options_10)
    para_color = np.zeros((model_v.r.shape[0],3))
    para_color[select_vtx_idx] = color_sum
    write_full_obj(model_v.r, model.f, np.array([]), np.array([]), np.array([]), np.array([]), para_color,
                   output_dir+'/debug_para_color.obj')
    print  coeff

    select_normal_list = []
    for i in range(0,len(select_vtx_idx)):
        adjacen_vtx_id = adjacency[i]
        if 1:
            # for j in range(0,len(adjacen_vtx_id)):
            adjacen_id1 = adjacen_vtx_id[0]
            vector1 = (model_v[adjacen_id1]- model_v[i])/ ch.linalg.norm((model_v[adjacen_id1]- model_v[i]))
            adjacen_id2 = adjacen_vtx_id[1]
            vector2 = (model_v[adjacen_id2]- model_v[i])/ ch.linalg.norm((model_v[adjacen_id2]- model_v[i]))
            #select_normal_list.append(ch.cross(vector1,vector2))
            #(x1, y1, z1)X(x2, y2, z2) = (y1z2 - y2z1, z1x2 - z2x1, x1y2 - x2y1)
            kk = ch.concatenate([vector1[1]*vector2[2]-vector2[1]*vector1[2],vector1[2]*vector2[0]-vector2[2]*vector1[0] ,vector1[0]*vector2[1]-vector2[0]*vector1[1]])
        # neibor_sum = ch.array([0,0,0])
        # for k in range(0,len(adjacen_vtx_id)):
        #     neibor_sum += model_v[adjacen_vtx_id[k]]
        # laplcian_normal = model_v[i]-neibor_sum
        laplcian_normal = kk
        select_normal_list.append( laplcian_normal/ch.linalg.norm(laplcian_normal) )
    #select_normal= ch.concatenate(select_normal_list)
    #print select_normal_list.shape
    #print select_normal_list[0].shape
    #检查法向的正确性
    if 1:
        v_color_caculate = np.zeros((len(select_vtx_idx),3),np.uint8)
        v_color_caculate[:,:] =[255,255,255]
        ray_dir = np.array([0,0,1])
        ray_dir = np.reshape(ray_dir,(3,1))
        select_normal_np = np.array([ x.r for x in select_normal_list ])
        dot_coef = np.dot(select_normal_np, ray_dir)
        dot_coef = np.array([ 0 if x <0.01 else x for x in dot_coef ])
        dot_coef = np.reshape(dot_coef,(dot_coef.shape[0],1))
        v_color_caculate = v_color_caculate*dot_coef
        #v_color_caculate = v_color_caculate*0.7+(1-0.7)*np.array([255,255,255])
    para_color = np.zeros((model_v.r.shape[0],3))
    para_color[select_vtx_idx] = v_color_caculate
    write_full_obj(model_v.r, model.f, np.array([]), np.array([]), np.array([]), np.array([]), para_color,
                   output_dir+'/debug_para_check_color.obj')

    # refection = ch.zeros(len(select_vtx_idx))
    # refection[:] =172
    refection = ch.array(color_sum)
    light = ch.array([1, 0, 0, 1])
    #求解光照参数
    r_vec =[]
    R_vec=[]
    for i in range(0,len(select_vtx_idx)):
        y_nij = ch.concatenate((ch.array([1]), select_normal_list[i][0], select_normal_list[i][1], select_normal_list[i][2]))  # [1,nx,ny,nz]
        r = refection[i] * ch.dot(light, y_nij)
        vtx = model_v.r[select_vtx_idx[i], :]
        x = int(vtx[0])
        y = int(vtx[1])
        R_vec.append( ori_img[y,x,::-1]) #要转化为 rgb
        r_vec.append(r)
        pass
    Edata = ch.concatenate(r_vec) - ch.concatenate(R_vec)

    objectives_pose ={}
    objectives_pose.update(
        {'face_lmk': Edata})
    # ch.minimize(fun=objectives_pose,
    #             x0=[light[1:4]], #, model.pose[3:6]
    #             method='dogleg',
    #             callback=None,
    #             options=opt_options_10)
    print light,
    #r_vec = r_vec.r
    r_vec = np.array([x.r for x in r_vec])
    para_color = np.zeros((model_v.r.shape[0],3))
    para_color[select_vtx_idx] = r_vec
    write_full_obj(model_v.r, model.f, np.array([]), np.array([]), np.array([]), np.array([]), para_color,
                   output_dir+'/debug_para_color_light.obj')

    r_vec =[]
    R_vec=[]
    for i in range(0,len(select_vtx_idx)):
        y_nij = ch.concatenate((ch.array([1]), select_normal_list[i][0], select_normal_list[i][1], select_normal_list[i][2]))  # [1,nx,ny,nz]
        r = refection[i] * ch.dot(light, y_nij)
        vtx = model_v.r[select_vtx_idx[i], :]
        x = int(vtx[0])
        y = int(vtx[1])
        R_vec.append( ori_img[y,x,::-1])
        r_vec.append(r)
        pass
    print type(ch.concatenate(r_vec))
    print type(ch.concatenate(R_vec))
    Edata = ch.concatenate(r_vec) - ch.concatenate(R_vec)
    print Edata.shape
    # regularizer
    # weights
    weights = {}
    weights['lmk'] = 1.0
    weights['shape'] = 0.001
    weights['expr'] = 0.001
    weights['pose'] = 0.01
    shape_num = 300
    expr_num = 100
    shape_idx = np.arange(0, min(300, shape_num))  # valid shape component range in "betas": 0-299
    expr_idx = np.arange(300, 300 + min(100, expr_num))  # valid expression component range in "betas": 300-399
    shape_err = weights['shape'] * model.betas[shape_idx]
    expr_err = weights['expr'] * model.betas[expr_idx]
    pose_err = weights['pose'] * model.pose[6:9]  # exclude global rotation
    objectives_pose ={}
    objectives_pose.update(
        {'face_lmk': Edata,  'shape': shape_err, 'expr': expr_err,
         'pose': pose_err})

    def on_step1(_):
        print 'call back'
        global g_count
        write_full_obj(model_v.r, model.f, np.array([]), np.array([]), np.array([]), np.array([]), v_color,
                       output_dir + '/debug_model_'+str(g_count)+'.obj')
        g_count+=1

    ch.minimize(fun=objectives_pose,
                x0=[model.betas[10:20]], #, model.pose[3:6]
                method='dogleg',
                callback=on_step1,
                options=opt_options_10)

    write_full_obj(model_v.r, model.f, np.array([]), np.array([]), np.array([]), np.array([]), v_color,
                   output_dir+'/debug_model_2.obj')

def extract_detail(subdived_mesh_path,ori_img,seg_img,output_dir):
    from fitting.util import build_adjacency,get_vertex_normal,sample_intensity_from_img,mat_save
    import cv2
    import math
    import geometry as geom
    from shortfunc import  adjmat_to_laplacian,mldivide
    import scipy.sparse as sparse
    from scipy.sparse.linalg import inv
    ori_img[:,:] = ori_img[::-1,:]
    intensity = cv2.cvtColor(ori_img,cv2.COLOR_RGB2GRAY)
    seg_img[:,:] = seg_img[::-1,:] # 转化为左下角
    subdived_mesh, subdived_mesh_f, t_frame_aligned, t_f_frame_aligned, n_frame_aligned, n_f_frame_aligned = read_igl_obj(
        subdived_mesh_path)
    vetex_noraml = get_vertex_normal(subdived_mesh,subdived_mesh_f)
    v_intensity = sample_intensity_from_img(subdived_mesh,vetex_noraml,intensity)
    adjmat = geom.adjmat_from_mesh(subdived_mesh.shape[0], subdived_mesh_f)
    adjmat = sparse.csc_matrix(adjmat)
    vtx_num = subdived_mesh.shape[0]
    L = adjmat_to_laplacian(adjmat)
    E = sparse.lil_matrix((vtx_num, vtx_num))
    ii = np.arange(vtx_num)
    E[ii, ii] = 1
    E = sparse.csc_matrix(E)

    # lp coordinate
    #lpc = L.dot(pts)


    select_vtx_idx =[]
    v_color = np.zeros((subdived_mesh.shape[0],3),np.uint8)
    for i in range(0,subdived_mesh.shape[0]):
        vtx = subdived_mesh[i,:]
        x = int(vtx[0])
        y = int(vtx[1])
        if x > seg_img.shape[1]-1:
            continue
        if y >  seg_img.shape[0]-1:
            continue
        if seg_img[y,x] :
            select_vtx_idx.append(i)
            v_color[i,:] = [255,0,0]
    Laplacian_matrix  =L
    dt = 0.2
    I_matrix =E
    r_v= v_intensity
    intensity_color = np.hstack((v_intensity,v_intensity,v_intensity))
    write_full_obj(subdived_mesh,subdived_mesh_f,np.array([]),np.array([]),np.array([]),np.array([]),intensity_color,output_dir+'/mesh_with_color.obj')
    #尝试把 顶点颜色生成图片
    def get_img_from_vertex(vertex,vertex_color,select_vtx_idx,width,height):
        if vertex.shape[0] != vertex_color.shape[0]:
            print 'input vertex row not equal to vertex color'
        result = np.zeros((height,width,vertex_color.shape[1]),np.uint8)
        z_buffer = np.zeros((height, width), np.float)
        z_buffer[:,:]=-10000
        for i in range(0,vertex.shape[0]):
            if i in select_vtx_idx:
                pass
            else:
                continue
            x =vertex[i,0]
            y = vertex[i,1]
            z = vertex[i,2]

            x =int(round(x))
            y=int(round(y))
            if x < 0:
                x = 0
            if y < 0:
                y = 0
            if x > result.shape[1] - 1:
                x = result.shape[1] - 1
            if y > result.shape[0] - 1:
                y = result.shape[0] - 1
            if z > z_buffer[y,x]:
                z_buffer[y, x] =z
                result[y,x] = vertex_color[i]
        result[:,:,:] = result[::-1,:,:]
        return result

    intensity_img =get_img_from_vertex(subdived_mesh,r_v,select_vtx_idx,ori_img.shape[1],ori_img.shape[0])
    cv2.imwrite(output_dir+'/intensity_img.png',intensity_img)

    from time import time
    timer_start = time()
    TMP = I_matrix - dt * Laplacian_matrix
    resut = {'TMP':TMP}
    #mat_save(resut,output_dir+'/tmp.mat')
    piu_v =mldivide(TMP,r_v)
    r_v = r_v[:,0]
    piu_v = r_v-piu_v
    timer_end = time()
    print "step1 %f sec\n" % (timer_end - timer_start)
    piu_v = piu_v.reshape((piu_v.shape[0],1))
    piu_img =get_img_from_vertex(subdived_mesh,piu_v,select_vtx_idx,ori_img.shape[1],ori_img.shape[0])
    cv2.imwrite(output_dir+'/piu_img'+str(dt)+'.png',piu_img)
    #return

    #Ain = inv(TMP)
    #piu_v = r_v -Ain*r_v
    #piu_v= piu_v[:,0]
    delta_piu_v = np.zeros( subdived_mesh.shape[0])
    from time import time
    timer_start = time()
    adjacency = build_adjacency(subdived_mesh, subdived_mesh_f)
    print "build_adjacency %f sec\n" % (timer_end - timer_start)
    from time import time
    timer_start = time()
    for i in range(0, subdived_mesh.shape[0] ):
        if i not in select_vtx_idx:
            continue
        adjacent_list = adjacency[i]
        adjacent_list =list(set(adjacent_list))
        alpha_v =0.0
        for neigbour_idx in adjacent_list:
            alpha_v+= math.exp( -np.linalg.norm(subdived_mesh[i,:]- subdived_mesh[neigbour_idx,:] ))
        up_sum =0.0
        for neigbour_idx in adjacent_list:
            tmp =np.dot( (subdived_mesh[i,:]-subdived_mesh[neigbour_idx,:]),vetex_noraml[i,:])/np.linalg.norm(subdived_mesh[i,:]-subdived_mesh[neigbour_idx,:])
            tmp = np.linalg.norm(tmp)
            up_sum+=math.exp(-np.linalg.norm(subdived_mesh[i, :] - subdived_mesh[neigbour_idx, :]))*(piu_v[i]-piu_v[neigbour_idx])*(1-tmp)
        delta_piu_v[i] = up_sum/alpha_v
    print "for %f sec\n" % (timer_end - timer_start)
    delta_piu_v = delta_piu_v.reshape((delta_piu_v.shape[0],1))
    #k =vetex_noraml * delta_piu_v[:]
    subdived_mesh= subdived_mesh+vetex_noraml*delta_piu_v[:]
    write_full_obj(subdived_mesh,subdived_mesh_f,np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),output_dir+'/detail_mesh'+str(dt)+'.obj')
    #print 'delta_piu_v',delta_piu_v

def extract_detailnew(input_obj,output_dir,objname,dt1 = -200000):
    from fitting.util import build_adjacency,get_vertex_normal,sample_intensity_from_img,mat_save
    import cv2
    import math
    import geometry as geom
    from shortfunc import  adjmat_to_laplacian2,mldivide,adjmat_to_laplacian
    import scipy.sparse as sparse
    subdived_mesh, subdived_mesh_f, t_frame_aligned, t_f_frame_aligned, n_frame_aligned, n_f_frame_aligned = read_igl_obj(
        input_obj)
    from triangle_raster import BBox3f
    bbox = BBox3f()
    bbox.addvertex_array(subdived_mesh)
    width = bbox.max[0] - bbox.min[0]
    height = bbox.max[1] - bbox.min[1]
    scale2 = width if width >height else height
    subdived_mesh/=scale2


    vcolor = readVertexColor(input_obj)
    vetex_noraml = get_vertex_normal(subdived_mesh, subdived_mesh_f)

    adjmat = geom.adjmat_from_mesh(subdived_mesh.shape[0], subdived_mesh_f)
    adjmat = sparse.csc_matrix(adjmat)
    vtx_num = subdived_mesh.shape[0]
    L = adjmat_to_laplacian2(adjmat)
    E = sparse.lil_matrix((vtx_num, vtx_num))
    ii = np.arange(vtx_num)
    E[ii, ii] = 1
    E = sparse.csc_matrix(E)

    Laplacian_matrix  =L
    dt = dt1
    I_matrix =E
    Intensity = np.zeros((vcolor.shape[0],1))
    for i in range(0,vcolor.shape[0]):
        Intensity[i]=vcolor[i,0]*0.3+vcolor[i,1]*0.59+vcolor[i,2]*0.11
    r_v= Intensity
    from time import time
    timer_start = time()
    TMP = I_matrix - dt * Laplacian_matrix
    resut = {'TMP': TMP}
    # mat_save(resut,output_dir+'/tmp.mat')
    piu_v = mldivide(TMP, r_v)
    r_v = r_v[:, 0]
    piu_v = r_v - piu_v
    timer_end = time()
#    print "step1 %f sec\n" % (timer_end - timer_start)

    delta_piu_v = np.zeros( subdived_mesh.shape[0])
    adjacency = build_adjacency(subdived_mesh, subdived_mesh_f)
#    print "build_adjacency %f sec\n" % (timer_end - timer_start)
    from time import time
    timer_start = time()
    for i in range(0, subdived_mesh.shape[0]):

        adjacent_list = adjacency[i]
        adjacent_list = list(set(adjacent_list))
        alpha_v = 0.0
        for neigbour_idx in adjacent_list:
            alpha_v += math.exp(-np.linalg.norm(subdived_mesh[i, :] - subdived_mesh[neigbour_idx, :]))
        up_sum = 0.0
        for neigbour_idx in adjacent_list:
            tmp = np.dot((subdived_mesh[i, :] - subdived_mesh[neigbour_idx, :]), vetex_noraml[i, :]) / np.linalg.norm(
                subdived_mesh[i, :] - subdived_mesh[neigbour_idx, :])
            tmp = np.linalg.norm(tmp)
            up_sum += math.exp(-np.linalg.norm(subdived_mesh[i, :] - subdived_mesh[neigbour_idx, :])) * (
            piu_v[i] - piu_v[neigbour_idx]) * (1 - tmp)
        delta_piu_v[i] = up_sum / alpha_v
#    print "for %f sec\n" % (timer_end - timer_start)
    delta_piu_v = delta_piu_v.reshape((delta_piu_v.shape[0], 1))
    # k =vetex_noraml * delta_piu_v[:]
#    print 'min',np.min(delta_piu_v)
#    print 'max',np.max(delta_piu_v)
    threshold =0.01
    # for i in range(0,delta_piu_v.shape[0]):
    #     if abs(delta_piu_v[i]) <threshold:
    #         delta_piu_v[i] =0.0
    scale =3.0
    subdived_mesh = subdived_mesh + vetex_noraml * delta_piu_v[:]*scale
    subdived_mesh*=scale2
    write_full_obj(subdived_mesh, subdived_mesh_f, np.array([]), np.array([]), np.array([]), np.array([]), vcolor,
                   output_dir + '/'+objname+'_detail_mesh_threshod_'+str(threshold) +' '+ str(dt) + 'scale_'+str(scale)+'.obj')


def fit_lmk3d_2(target_v,target_f,target_lmk_3d_idx,  # input landmark 3d
              model,  # model
              mesh_faces,
              lmk_facevtx_idx,  # landmark embedding
              lmk_face_idx,
              lmk_b_coords,
              weights,  # weights for the objectives
              shape_num=300, expr_num=100, opt_options=None,output_dir=None):
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
    from fitting.landmarks import landmark_error_3d_only,p2perror
    import chumpy as ch
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

    g_scale = ch.array([1800])
    g_trans_3d = ch.array([200, 300, 0])
    optim_expression = 0
    print "\nstep 1: 根据face landmark 来获得3维位移和放缩,..."
    timer_start = time()
    face_lmk_err = landmark_error_3d_only(
        scale =g_scale,
        trans_3d= g_trans_3d,
        mesh_verts= model,
        mesh_faces=mesh_faces,
        lmk_face_idx=lmk_face_idx,
        lmk_b_coords=lmk_b_coords,
        target_v = target_v,
        targrt_face= target_f,
        target_lmk_idx=target_lmk_3d_idx,
        weight=1)


    objectives_pose = {}
    objectives_pose.update({'face_lmk': face_lmk_err})

    ch.minimize(fun=objectives_pose,
                x0=[g_scale, g_trans_3d,model.pose[0:3]],
                method='dogleg',
                callback=on_step,
                options=opt_options_50)
    timer_end = time()

    #    output_dir = 'L:/yuanqing/imgs/imgs/vrn_result/niutou/10/'
    def print_para_resut(step, timer_end, timer_start):
        print step
        print "in %f sec\n" % (timer_end - timer_start)
        print "model.trans"
        print model.trans
        print "all pose"
        print model.pose[:]
        print "g_scale"
        print g_scale
        print "g_trans_3d"
        print g_trans_3d

    from  para_texture_buid import add_texture_and_write
    def debug_result(output_path, filename):
        #return
        output_path = output_path + filename
        g_v = g_scale.r * model.r[:, :]
        g_v[:, :] = g_v[:, :] + g_trans_3d.r
        #write_simple_obj(mesh_v=g_v, mesh_f=model.f, filepath=output_path, verbose=False)
        frame_re_texture_map = 'D:\mproject/face-reconstruct/texpc\source_para/texture/frame_template_retex.obj'
        v_frame_aligned, f_frame_aligned, t_frame_aligned, t_f_frame_aligned, n_frame_aligned, n_f_frame_aligned = read_igl_obj(
            frame_re_texture_map)
        write_full_obj(g_v,model.f, n_frame_aligned, n_f_frame_aligned, t_frame_aligned,
                       t_f_frame_aligned, np.array([]), output_path, generate_mtl=True, img_name='mean_tex_color.png')
        #add_texture_and_write(v_np=g_v, f_np=model.f, output_path=output_path)

    print_para_resut('step1', timer_end, timer_start)
    debug_result(output_dir, 'step_2d_1.obj')

    print "\nstep 2: 根据body landmark 来获得2维位移和旋转..."
    face_lmk_err = landmark_error_3d_only(
        scale =g_scale,
        trans_3d= g_trans_3d,
        mesh_verts= model,
        mesh_faces=mesh_faces,
        lmk_face_idx=lmk_face_idx,
        lmk_b_coords=lmk_b_coords,
        target_v = target_v,
        targrt_face= target_f,
        target_lmk_idx=target_lmk_3d_idx,
        weight=1)
    objectives_pose.update({'face_lmk': face_lmk_err})
    if optim_expression:
        ch.minimize(fun=objectives_pose,
                    x0=[g_scale, g_trans_3d,model.pose[0:3],model.pose[6:9]],
                    method='dogleg',
                    callback=on_step,
                    options=opt_options_20)
    else:
        ch.minimize(fun=objectives_pose,
                    x0=[g_scale, g_trans_3d,model.pose[0:3]],
                    method='dogleg',
                    callback=on_step,
                    options=opt_options_20)
    print_para_resut('step2', timer_end, timer_start)
    debug_result(output_dir, 'step_2d_2.obj')
    print "\nstep 3: update g_scale,g_trans_2d,model.pose[0:6],model.betas ..."
    # regularizer
    shape_err = weights['shape'] * model.betas[shape_idx]
    expr_err = weights['expr'] * model.betas[expr_idx]
    pose_err = weights['pose'] * model.pose[6:9]  # exclude global rotation

    timer_start = time()
    face_lmk_err = landmark_error_3d_only(
        scale =g_scale,
        trans_3d= g_trans_3d,
        mesh_verts= model,
        mesh_faces=mesh_faces,
        lmk_face_idx=lmk_face_idx,
        lmk_b_coords=lmk_b_coords,
        target_v = target_v,
        targrt_face= target_f,
        target_lmk_idx=target_lmk_3d_idx,
        weight=1/g_scale.r*0.8)
    objectives_pose.update({'face_lmk': face_lmk_err, 'shape': shape_err, 'expr': expr_err,
         'pose': pose_err})
    if optim_expression:
        ch.minimize(fun=objectives_pose,
                    x0=[model.pose[6:9],model.betas[0:400]],
                    method='dogleg',
                    callback=on_step,
                    options=opt_options_20)
    else:
        ch.minimize(fun=objectives_pose,
                    x0=[model.betas[0:300]],
                    method='dogleg',
                    callback=on_step,
                    options=opt_options_20)
    timer_end = time()
    print_para_resut('step3', timer_end, timer_start)
    debug_result(output_dir, 'step_2d_3.obj')

    print "\nstep 4: model.betas ,p2p"
    timer_start = time()
    import fitting.global_var as global_var
    from fitting.util import  read_int
    frame_front_mask = read_int('D:\mproject/face-reconstruct' + '/face_front_mask.txt')
    frame_front_mask= frame_front_mask[:, 0]
    face_mask = model.r[frame_front_mask]
    write_landmark_to_obj(output_dir+'/front_mask1.obj',face_mask,size = 1)
    for i in range(0,20):
        frame_front_mask = np.array(range(0,model.r.shape[0]))
        p2p_error = p2perror(
            scale=g_scale, trans_2_3d=g_trans_3d, mesh_verts=model, mesh_faces=mesh_faces,
            target_3d_v =target_v, target_3d_f=target_f,mask_facevtx_idx = frame_front_mask, p2p_weight=1/g_scale.r
        )
        objectives_pose ={}
        objectives_pose.update({'p2p_error':p2p_error,'face_lmk': face_lmk_err, 'shape': shape_err, 'expr': expr_err, 'pose': pose_err})
        if optim_expression:
            ch.minimize( fun      = objectives_pose,
                         x0       = [model.betas[ 0:400]],
                         method   = 'dogleg',
                         callback = on_step,
                         options  = opt_options_20 )
        else:
            ch.minimize( fun      = objectives_pose,
                         x0       = [model.betas[ 0:300]],
                         method   = 'dogleg',
                         callback = on_step,
                         options  = opt_options_20 )
        if i%5 ==0:
            print_para_resut('step4', timer_end, timer_start)
            debug_result(output_dir, 'step_2d_'+'iter_'+str(i)+'.obj')
    debug_result(output_dir, 'step_2d_' + 'iter_' + str(i) + '.obj')
    face_mask = model.r[frame_front_mask]
    write_landmark_to_obj(output_dir+'/front_mask2.obj',face_mask,size = 1)

    global_rotate_pose = model.pose[0:3]
    print  global_rotate_pose
    import cv2
    global_rotate_pose = np.array([global_rotate_pose[0].r, global_rotate_pose[1].r, global_rotate_pose[2].r])
    print  global_rotate_pose
    global_rotate = cv2.Rodrigues(global_rotate_pose)[0]
    print global_rotate
    parms = {'trans': g_trans_3d.r, 'pose': model.pose.r, 'betas': model.betas.r,'scale':g_scale.r,'global_rotate':global_rotate}
    print parms
    g_v = g_scale.r * model.r[:, :]
    g_v[:, :] = g_v[:, :] + g_trans_3d.r
    return g_v, model.f, parms


'''
img 只是拿它的高和宽
'''
def get_z_buffer(V,F,img,output_render_img,vcolor = np.array([])):
    sys.path.insert(0, "D:/mproject/meshlab2016/meshlab/src/x64/Release/")
    import meshlab_python
    height = img.shape[0]
    width = img.shape[1]
    max_z = max(V[:,2])
    min_z = min(V[:, 2])
    bbox_list = [int(0), int(0), int(-600), int(width),
                 int(height), int(600)]
    if vcolor.size ==0:
        color = np.zeros((V.shape[0],3),np.uint8)
        color[:,:] =[255,255,255]
    else:
        color = vcolor.astype(int)
    result = meshlab_python.Mesh_render_to_image_withmy_bbox(output_render_img,
                                                             V.tolist(), F.tolist(), [], [], [], [],
                                                             color.tolist(),
                                                             int(width), int(height), bbox_list)
    render_img = np.array(result[0])
    z_buffer_img = np.array(result[1])
    return render_img,z_buffer_img


    pass


def shape_from_shading_detail(height_map,img,seg_img,boundary_pixel):
    print 'start shape_from_shading_detail'
    height_map = height_map[::-1,:]
    img = img[::-1,:]
    seg_img = seg_img[::-1,:]
    boundary_pixel[:,1] = height_map.shape[0]-1 - boundary_pixel[:,1]
    # height map ,img,seg img 应具有相同的长宽
    import chumpy as ch
    #z_buffer
    #render_img, height_map = get_z_buffer(V, F,img)
    height_map_varibales = ch.zeros((height_map.shape[0],height_map.shape[1]),np.float)
    valid_map = np.zeros((height_map.shape[0],height_map.shape[1]),np.bool)
    for j in range(0,height_map.shape[0]):
        for i in range(0,height_map.shape[1]):
            if height_map[j,i]>1 and seg_img[j,i]:
                height_map_varibales[j,i] = height_map[j,i]
                valid_map[j,i] = True
    normal_map_list=[]
    for y in range(0,300):#range(0,height_map.shape[0]):
        for x in range(0,200): #range(0,height_map.shape[1]):
            if valid_map[y,x] and valid_map[y,x+1] and valid_map[y+1,x]:
                # hxy= ch.concatenate((x,y,height_map_varibales[y,x]))
                # hx_1y = ch.concatenate((x+1, y, height_map_varibales[y, x]))
                # hxy_1 = ch.concatenate((x, y+1, height_map_varibales[y, x]))
                # normal = ch.cross( hx_1y-hxy,hxy_1-hxy)
                # normal2 = normal/ch.linalg.norm(normal)
                normal_map_list.append(ch.array([0, 0, 1]))
                #normal_map_list.append(normal2)
                print y,x
            else:
                 normal_map_list.append(ch.array([0,0,1]))
    normal_map_1 = ch.concatenate((normal_map_list))
    print 'normal_map_1.shape',normal_map_1.shape
    return
    normal_map =normal_map_1.reshape((height_map.shape[0],height_map.shape[1],3))
    print normal_map[0, 0]
    R_vec =[]
    r_vec =[]
    shder_vec =[]
    #假设只考虑一个通道
    refection = ch.zeros((height_map.shape[0],height_map.shape[1]),np.float)
    refection[:,:] = 172 #[172,191,218]
    light = ch.array([1.0,0.0,0.0,-1.0])
    #1 .data 项
    print 'construct data term'
    render_map_list = []
    for y in range(0,height_map.shape[0]):
        for x in range(0,height_map.shape[1]):
            if valid_map[y,x]:
                print normal_map[y,x,0]
                y_nij = ch.concatenate((ch.array([1]),normal_map[y,x,0],normal_map[y,x,1],normal_map[y,x,2]))   # [1,nx,ny,nz]
                r= refection[y,x]*ch.dot( light,y_nij)
                R_vec.append(img[y,x,0])
                r_vec.append(r)
                render_map_list.append(r)
            else:
                render_map_list.append(0)
    render_map1 = ch.concatenate(render_map_list)
    render_map = render_map1.reshape((height_map.shape[0],height_map.shape[1]))

    E_data = ch.concatenate(R_vec) - ch.concatenate(r_vec)
    #2 . 梯度相似项
    print 'construct gradient term'
    R_gradient =[]
    r_gradient =[]

    for y in range(0,height_map.shape[0]):
        for x in range(0,height_map.shape[1]):
            if valid_map:
                rij= render_map[j,i]
                ri_1j = render_map[j, i+1]
                rij_1 = render_map[j+1, i]
                Rij = img[y,x,0]
                Ri_1j = img[y, x+1, 0]
                Rij_1 = img[y+1, x, 0]
                R_gradient.append(Ri_1j- Rij)
                R_gradient.append(Rij_1 - Rij)
                r_gradient.append(ri_1j-rij)
                r_gradient.append(rij_1 - rij)

    E_gadient = ch.concatenate(R_gradient)-ch.concatenate(r_gradient)
    #3 法向相似项

    #4.法向平滑项
    print 'construct smooth term'
    normal_gredinet1=[]
    normal_gredinet2 = []
    for y in range(0,height_map.shape[0]):
        for x in range(0,height_map.shape[1]):
            if valid_map:
                nij= normal_map[j,i]
                ni_1j = normal_map[j, i+1]
                nij_1 = normal_map[j+1, i]
                normal_gredinet1.append(ni_1j)
                normal_gredinet1.append(nij_1)
                normal_gredinet2.append(nij)
                normal_gredinet2.append(nij)

    Esmooth = ch.concatenate(normal_gredinet1)-ch.concatenate(normal_gredinet2)

    all_Energy = {'E_data':E_data,'E_gadient':E_gadient,'Esmooth':Esmooth}
    # on_step callback
    def on_step(_):
        print 'call back'
    print 'start minimize'
    ch.minimize(fun=all_Energy,
                x0=[height_map_varibales[valid_map]],
                method='dogleg',
                callback=on_step,
                options=opt_options_50)


def test_pncc():
    input_mesh_path = 'E:\workspace\dataset\hairstyles/templat_xyz.obj'
    v_frame_aligned, f_frame_aligned, t_frame_aligned, t_f_frame_aligned, n_frame_aligned, n_f_frame_aligned = read_igl_obj(
        input_mesh_path)
    mean_v = np.mean(v_frame_aligned,axis=0)
    max_x = max(v_frame_aligned[:,0])
    min_x= min(v_frame_aligned[:,0])
    mean_x = (max_x + min_x)/2
    scale_x = 1/((max_x -min_x)/2)
    max_y = max(v_frame_aligned[:,1])
    min_y= min(v_frame_aligned[:,1])
    mean_y = (max_y + min_y) / 2
    scale_y = 1/((max_y - min_y) / 2)
    max_z = max(v_frame_aligned[:,2])
    min_z= min(v_frame_aligned[:,2])
    mean_z = (max_z + min_z) / 2
    scale_z = 1/((max_z - min_z) / 2)
    pncc= (v_frame_aligned -np.array([mean_x,mean_y,mean_z]))*np.array([scale_x,scale_y,scale_z])
    print 'mean_x,mean_y,mean_z,scale_x,scale_y,scale_z',mean_x,mean_y,mean_z,scale_x,scale_y,scale_z
    write_full_obj(pncc,f_frame_aligned,np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),'E:\workspace\dataset\hairstyles/test_pncc.obj')
def test_scale_to_pncc():
    input_mesh_path = 'E:\workspace\dataset\hairstyles/test_pncc/step_2d_iter_20.obj'
    v_frame_aligned, f_frame_aligned, t_frame_aligned, t_f_frame_aligned, n_frame_aligned, n_f_frame_aligned = read_igl_obj(
        input_mesh_path)
    para = np.array([    -0.0015395,0.0165155,0.0774565,1.46265227124,1.08692639497,2.6516582808])
    mean_x = para[0]
    mean_y = para[1]
    mean_z = para[2]
    scale_x = para[3]
    scale_y = para[4]
    scale_z = para[5]
    frame_pncc = (v_frame_aligned -np.array([mean_x,mean_y,mean_z]))* np.array([scale_x,scale_y,scale_z])
    color =[]
    for i in range(0,frame_pncc.shape[0]):
        xyz =frame_pncc[i,:]
        if xyz[0] <=1.0 and xyz[1] <=1.0 and xyz[2]<=1.0 and xyz[0]>=-1.0 and xyz[1] >=-1.0 and xyz[2] >=-1.0:
            r = int(255.0*(xyz[0]+1.0)/2)
            g = int(255.0 * (xyz[1] + 1.0)/2)
            b = int(255.0 * (xyz[2] + 1.0)/2)
            color.append([r,g,b])
            pass
        else:
            color.append([0,0,0])
    color = np.array(color)
    #write_full_obj(frame_pncc, f_frame_aligned, np.array([]), np.array([]), np.array([]), np.array([]), color,
    #               'E:\workspace\dataset\hairstyles/frame_pncc.obj')
    sys.path.insert(0, "D:/mproject/meshlab2016/meshlab/src/x64/Release/")
    input_mesh_path ='E:/workspace/dataset/hairstyles/2d_hair/result/A13010436665609/builded_hair/frame_aligned_to_image.obj'
    input_dir = 'E:/workspace/dataset/hairstyles/2d_hair/result/A13010436665609/builded_hair/'
    seg_image_path = 'E:/workspace/dataset/hairstyles/2d_hair/result/A13010436665609/builded_hair/A13010436665609.png'
    ori_image_path = 'E:/workspace/dataset/hairstyles/2d_hair/A13010436665609.jpg'
    import meshlab_python
    v_frame_aligned, f_frame_aligned, t_frame_aligned, t_f_frame_aligned, n_frame_aligned, n_f_frame_aligned = read_igl_obj(input_mesh_path)


    bbox_2d= [30,140,400,520]
    bbox_list = [int(bbox_2d[0]), int(bbox_2d[1]), int(-600), int(bbox_2d[2]),
                 int(bbox_2d[3]), int(600)]
    result = meshlab_python.Mesh_render_to_image_withmy_bbox('E:/workspace/dataset/hairstyles/2d_hair/result/A13010436665609/builded_hair/get_z_buffer.png',
                                                             v_frame_aligned.tolist(), f_frame_aligned.tolist(), [], [], [], [],
                                                             color.tolist(),
                                                             int(512), int(512), bbox_list)
    render_img = np.array(result[0])
    render_img = render_img.astype(float)
    z_buffer_img = np.array(result[1])
    import cv2
    oir_img = cv2.imread(ori_image_path ,cv2.IMREAD_COLOR)

    from  fitting.util import rescale_imge_with_bbox

    oir_img = oir_img[::-1,:,:]
    oir_img = rescale_imge_with_bbox(oir_img,bbox_2d)
    oir_img = oir_img[::-1, :, :]
    oir_img = cv2.resize(oir_img, (int(512), int(512)), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite('E:/workspace/dataset/hairstyles/test_pncc/oir_img.png', oir_img)
    seg_img_binary = cv2.imread(input_dir+'A13010436665609_face_2.png', cv2.IMREAD_GRAYSCALE)
    seg_img_binary = seg_img_binary.reshape((seg_img_binary.shape[0],seg_img_binary.shape[1],1))
    seg_img_binary = rescale_imge_with_bbox(seg_img_binary[::-1,:,:],bbox_2d)



    seg_img_binary = seg_img_binary[::-1,:,:]
    seg_img_binary = cv2.resize(seg_img_binary, (int(512), int(512)), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite('E:/workspace/dataset/hairstyles/test_pncc/seg_img_binary.png', seg_img_binary)
    seg_img_binary = seg_img_binary>1
    for j in range(0, z_buffer_img.shape[0]):
        for i in range(0, z_buffer_img.shape[1]):
            if not seg_img_binary[j, i] or z_buffer_img[j, i] < 3:
                z_buffer_img[j, i] = 0
                render_img[j, i] = [0,0,0]
                pass
    cv2.imwrite('E:/workspace/dataset/hairstyles/test_pncc/render_img.png',render_img[:,:,::-1])
    cv2.imwrite('E:/workspace/dataset/hairstyles/test_pncc/z_buffer_img.png',z_buffer_img)
    depth_img = np.zeros((512,512,3))
    for j in range(0,z_buffer_img.shape[0]):
        for i in range(0,z_buffer_img.shape[1]):
            depth_img[j, i, 0] = i/2.0
            depth_img[j, i, 1] = j/2.0
            depth_img[j,i,2] = z_buffer_img[j,i]

    Target = {'im_depth': depth_img,'im_pncc':render_img}
    depth_img = depth_img.reshape((512*512,3))
    depth_img[:, 1] = 512 -depth_img[:,1]
    write_full_obj(depth_img/512,np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]),
                   'E:/workspace/dataset/hairstyles/test_pncc/depth_pointcloud.obj')
    mat_save(Target, 'E:/workspace/dataset/hairstyles/test_pncc/' + 'all.mat')
def test_scale_to_ori():
    para = np.array([-0.0015395, 0.0165155, 0.0774565, 1.46265227124, 1.08692639497, 2.6516582808])
    input_mesh_path = 'E:\workspace\dataset\hairstyles/test_pncc/final_refine_obj.obj'
    v_frame_aligned, f_frame_aligned, t_frame_aligned, t_f_frame_aligned, n_frame_aligned, n_f_frame_aligned = read_igl_obj(
        input_mesh_path)
    color = readVertexColor(input_mesh_path)
    mean_x = para[0]
    mean_y = para[1]
    mean_z = para[2]
    scale_x = para[3]
    scale_y = para[4]
    scale_z = para[5]
    pncc = (v_frame_aligned - np.array([mean_x, mean_y, mean_z])) * np.array([scale_x, scale_y, scale_z])
    pncc = pncc/2+0.5
    write_full_obj(pncc, f_frame_aligned, np.array([]), np.array([]), np.array([]), np.array([]), color,
                   'E:\workspace\dataset\hairstyles/test_pncc/final_refine_pncc_obj.obj')
    pass
def test_landmark():
    input_mesh = 'E:/workspace/dataset/hairstyles/frame_template_retex.obj'
    out_put_landmark ='E:/workspace/dataset/hairstyles/frame_template_retex_landmarks.obj'
    test_frame_mesh_landmark(input_mesh, out_put_landmark,size = 10)

def fitting_frame_to_mesh_by3d_landmark( target_v,target_f,target_landmark_idx,out_put_dir,
                   frame_model_path = './models/generic_model.pkl',frame_lmk_emb_path='./data/lmk_embedding_intraface_to_flame.pkl' ):
    # if os.path.exists(out_put_dir + '/face_result.pkl'):
    #     result = load_binary_pickle( filepath=out_put_dir + '/face_result.pkl')
    #     return  result

    from smpl_webuser.serialization import load_model
    from fitting.landmarks import load_embedding, landmark_error_3d
    model = load_model(frame_model_path)  # the loaded model object is a 'chumpy' object, check https://github.com/mattloper/chumpy for details
    # landmark embedding
    lmk_face_idx, lmk_b_coords = load_embedding(frame_lmk_emb_path)
    #68 个 landmark
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
    safe_mkdirs(out_put_dir)

    #landmark_v=target_landmark_idx


    # weights
    weights = {}
    weights['lmk'] = 1.0
    weights['shape'] = 0.001
    weights['expr'] = 0.001
    weights['pose'] = 0.01
    # default optimization options
    import scipy.sparse as sp
    opt_options = {}
    opt_options['disp'] = 1
    opt_options['delta_0'] = 0.1
    opt_options['e_3'] = 1e-4
    opt_options['maxiter'] = 10
    sparse_solver = lambda A, x: sp.linalg.cg(A, x, maxiter=opt_options['maxiter'])[0]
    opt_options['sparse_solver'] = sparse_solver

    mesh_v, mesh_f, parms = fit_lmk3d_2(target_v=target_v,  # input landmark 3d
                                        target_f = target_f,
                                        target_lmk_3d_idx=target_landmark_idx,
                                        model=model,  # model
                                        mesh_faces=model.f,
                                        lmk_facevtx_idx=face_select_lmk,  # landmark embedding
                                        lmk_face_idx=lmk_face_idx,
                                        lmk_b_coords=lmk_b_coords,
                                        weights=weights,  # weights for the objectives
                                        shape_num=300, expr_num=100, opt_options=opt_options,
                                        output_dir=out_put_dir)  # options

    face_result ={'mesh_v':mesh_v,'mesh_f':mesh_f,'parms':parms}
    output_path = out_put_dir+ '/fitting_frame_face.obj'
    frame_re_texture_map = 'D:\mproject/face-reconstruct/texpc\source_para/texture/frame_template_retex.obj'
    v_frame_aligned, f_frame_aligned, t_frame_aligned, t_f_frame_aligned, n_frame_aligned, n_f_frame_aligned = read_igl_obj(frame_re_texture_map)
    write_full_obj(mesh_v,mesh_f,n_frame_aligned,n_f_frame_aligned,t_frame_aligned,t_f_frame_aligned,np.array([]),output_path,generate_mtl=True,img_name='mean_tex_color.png')
    #write_simple_obj(mesh_v=mesh_v, mesh_f=mesh_f, filepath=output_path, verbose=False)
    save_binary_pickle(face_result, out_put_dir+'/fitting_frame_face.pkl')
    return  face_result

def generate_mesh_with_color():
    ori_img = cv2.imread('E:\workspace/vrn_data/bgBlue/A13010436665609/generate_face/'+'A1301074625120A.jpg',cv2.IMREAD_UNCHANGED)
    img_mask = cv2.imread('E:\workspace/vrn_data/bgBlue/A13010436665609/generate_face/' + 'A1301074625120A_seg.png',
                          cv2.IMREAD_GRAYSCALE)
    valid_img = np.zeros((img_mask.shape[0],img_mask.shape[1]),dtype=np.bool)
    for j in range(0,img_mask.shape[0]):
        for i in range(0,img_mask.shape[1]):
            if img_mask[j,i] >1:
                valid_img[j, i] = True
            else:
                valid_img[j,i] = False
    from  hair_orientation import  convert_img_2_mseh_new
    V,F,C = convert_img_2_mseh_new(ori_img,valid_img)
    V = np.array(V)
    F = np.array(F)
    C = np.array(C)
    V=V/512.0*2-1
    C=C/255.0
    # write_full_obj(V,F,np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),
    #                'E:/workspace/dataset/hairstyles/2d_hair/result/A1301043736980A/img_mask_.obj')
    write_full_obj(V,F,np.array([]),np.array([]),np.array([]),np.array([]),C,
                   'E:\workspace/vrn_data/bgBlue/A13010436665609/generate_face/A1301074625120A_with_color.obj')


def frame_head_optimaization(v,f):
    left_v =v[1100]
    right_v = v[305]
    center_v = (left_v+right_v)/2.0
    front_v = v[3555]
    top_v = v[3559]
    head_scale = np.linalg.norm(left_v - right_v)
    print np.linalg.norm(left_v-right_v)
    pre_head_scale=0.192929
    scale = head_scale/pre_head_scale
    from fitting.util import read_int,Harmonic
    fram_back_path = 'D:/mproject/face-reconstruct/texpc/frame_back2.txt'
    frame_back_idx = read_int(fram_back_path)
    frame_back_idx = frame_back_idx[:,0].tolist()
    all_index =range(0,v.shape[0])
    fix_landmark_idx = []
    mov_landmark_idx = [3539]
    for i in range(0,len(all_index)):
        if all_index[i] in frame_back_idx:
            continue
        fix_landmark_idx.append(all_index[i])
    b_landmark_idx = fix_landmark_idx+mov_landmark_idx
    b_landmark_idx = np.array(b_landmark_idx)
    fix_v = v[fix_landmark_idx,:]
    move_v = v[mov_landmark_idx]
    dir =np.array([0,0.01,-0.01])
    dir = dir/np.linalg.norm(dir)
    dir = move_v-center_v
    dir = dir / np.linalg.norm(dir)
    move_v = move_v-0.05*scale* dir
    bc_v = np.vstack((fix_v,move_v))
    w = Harmonic(v, f, b_landmark_idx, bc_v, k=2)
    opti_v=[]
    opti_f=[]
    pass
    return w

def extract_para_texture(input_texture_img,input_texture_seg,para_dir):
    import chumpy as ch
    input_texture_img = cv2.imread(input_texture_img, cv2.IMREAD_COLOR)
    img_height = input_texture_img.shape[0]
    img_width = input_texture_img.shape[1]
    input_texture_seg = cv2.imread(input_texture_seg, cv2.IMREAD_GRAYSCALE)
    input_texture_img = input_texture_img.reshape((img_height*img_width,3))
    input_texture_seg= input_texture_seg.reshape((img_height*img_width,1))
    select_idx =[]
    for i in range(0,input_texture_seg.shape[0]):
        if input_texture_seg[i,0]>125:
            select_idx.append(i)
    target_texture = input_texture_img[select_idx]
    target_texture = target_texture.astype(np.float)

    mean_img = cv2.imread(para_dir+'mean_tex_color.png',cv2.IMREAD_COLOR)
    mean_img= mean_img.reshape((img_height*img_width,3))
    select_mean_img = mean_img[select_idx]
    select_mean_img = select_mean_img.astype(float)
    print np.max(select_mean_img), np.min(select_mean_img)
    coeff_num =5
    select_tx_pc = np.zeros((coeff_num,len(select_idx),3))
    for i in range(0,coeff_num):
        para_img = cv2.imread(para_dir + 'pc_'+str(i).zfill(3)+'std_5+.png', cv2.IMREAD_COLOR)
        para_img = para_img.reshape((img_height*img_width,3))
        select_para_img = para_img[select_idx]
        select_para_img = select_para_img.astype(float)
        print np.max(select_para_img), np.min(select_para_img)
        select_tx_pc[i,:,:] = select_para_img - select_mean_img
    coeff = ch.zeros(coeff_num)
    color_sum = ch.zeros((len(select_idx),3))
    color_sum+=select_mean_img
    for p in range(0, coeff_num):
        color_sum+=coeff[p]*select_tx_pc[p,:,:]
    color_term = color_sum - target_texture
    color_term/=255.0
    para_input = 'D:/mproject/face-reconstruct/texpc/target_para/frame_tex.pkl'
    data = load_binary_pickle(para_input)
    texEV = data['texEV']
    objectives_pose ={}
    # objectives_pose.update(
    #     {'data term': color_term})
    objectives_pose.update(
        {'data term': color_term,  'reg': 1000*(1/texEV)*coeff})
    def on_step1(_):
        print 'call back'
    ch.minimize(fun=objectives_pose,
                x0=[coeff[0:coeff_num]], #, model.pose[3:6]
                method='dogleg',
                callback=on_step1,
                options=opt_options_20)
    print coeff[0:coeff_num]
    color_sum = np.zeros((img_height*img_width, 3))
    color_sum = color_sum+mean_img
    for i in range(0,coeff_num):
        para_img = cv2.imread(para_dir + 'pc_'+str(i).zfill(3)+'std_5+.png', cv2.IMREAD_COLOR)
        para_img = para_img.reshape((img_height*img_width,3))
        color_sum+=coeff[i].r*(para_img-mean_img)
    color_sum = color_sum.reshape((img_height,img_width,3))
    color_sum = np.clip(color_sum, 0, 255)
    print np.max(color_sum),np.min(color_sum)
    #color_sum = color_sum.astype(np.uint8)
    return color_sum

def extract_para_texture_no_chumpy(input_texture_img,input_texture_seg,para_dir):
    import numpy.linalg as LA
    input_texture_img = cv2.imread(input_texture_img, cv2.IMREAD_COLOR)
    img_height = input_texture_img.shape[0]
    img_width = input_texture_img.shape[1]
    input_texture_seg = cv2.imread(input_texture_seg, cv2.IMREAD_GRAYSCALE)
    input_texture_img = input_texture_img.reshape((img_height*img_width,3))
    input_texture_seg= input_texture_seg.reshape((img_height*img_width,1))
    select_idx =[]
    for i in range(0,input_texture_seg.shape[0]):
        if input_texture_seg[i,0]>125:
            select_idx.append(i)
    target_texture = input_texture_img[select_idx]
    pixel_num = len(select_idx)
#    target_texture = target_texture.astype(np.float)
    mean_img = cv2.imread(para_dir+'mean_tex_color.png',cv2.IMREAD_COLOR)
    mean_img= mean_img.reshape((img_height*img_width,3))
    select_mean_img = mean_img[select_idx]
#    select_mean_img = select_mean_img.astype(float)
    print np.max(select_mean_img), np.min(select_mean_img)
    coeff_num =1
    selected_blendshape = np.zeros((coeff_num, pixel_num, 3))
    for i in range(0,coeff_num):
        para_img = cv2.imread(para_dir + 'pc_'+str(i).zfill(3)+'std_5+.png', cv2.IMREAD_COLOR)
        para_img = para_img.reshape((img_height*img_width,3))
        select_para_img = para_img[select_idx]
#        select_para_img = select_para_img.astype(float)
        print np.max(select_para_img), np.min(select_para_img)
        #select_tx_pc[i,:,:] = select_para_img - select_mean_img
        selected_blendshape[i,:,:] = select_para_img - select_mean_img
    A = np.zeros([3 * pixel_num, coeff_num])
    b = np.zeros(3 * pixel_num)
    for i in xrange(pixel_num):
        h = target_texture[i,:] - select_mean_img[i,:]
        b[3*i+0] = h[0]
        b[3 * i + 1] = h[1]
        b[3 * i + 2] = h[2]
        for j in xrange(coeff_num):
            HH = selected_blendshape[j,i,:].reshape([3,1])
            A[3 * i+0, j] = HH[0,0]
            A[3 * i + 1, j] = HH[1,0]
            A[3 * i + 2, j] = HH[2,0]
    A = A/255.0
    b = b/255.0
    h4 = 10000
    h5 = 0.01
    para_input = 'D:/mproject/face-reconstruct/texpc/target_para/frame_tex.pkl'
    data = load_binary_pickle(para_input)
    texEV = data['texEV']
    B = h4* np.eye(coeff_num)
    C = h5*np.eye(coeff_num)
    for i in range(0,coeff_num):
        B[i,i] *=1/texEV[i]
    AA = A.T.dot(A) + B.T.dot(B)
    bb = A.T.dot(b)
    beta = LA.lstsq(AA, bb)[0]
    print beta


    # objectives_pose.update(
    #     {'data term': color_term,  'reg': (1/texEV)*coeff})

    color_sum = np.zeros((img_height*img_width, 3))
    color_sum = color_sum+mean_img
    for i in range(0,coeff_num):
        para_img = cv2.imread(para_dir + 'pc_'+str(i).zfill(3)+'std_5+.png', cv2.IMREAD_COLOR)
        para_img = para_img.reshape((img_height*img_width,3))
        color_sum+=beta[i]*(para_img-mean_img)
    color_sum = color_sum.reshape((img_height,img_width,3))
    print np.max(color_sum), np.min(color_sum)
    color_sum = np.clip(color_sum, 0, 255)
    print np.max(color_sum),np.min(color_sum)
    color_sum = color_sum.astype(np.uint8)
    return color_sum

def genetate_expression_single(shape_model ,vrn_object_dir,object_name,project_dir,out_put_dir,frame_model_path='./models/male_model.pkl',use_3d_landmark=False,preresult= None):
    if 0 and os.path.exists(project_dir + 'Landmark/' + object_name + '.txt'):
        pass
        hm_3d_path = project_dir + 'Landmark/' + object_name + '.txt'
    else:
        hm_3d_path = vrn_object_dir + '/2d/' + object_name + '.txt'
    oriimage_file_path = vrn_object_dir + '/' + object_name + '.jpg'
    ori_image = readImage(oriimage_file_path)
    height, width, dim = ori_image.shape
    landmark_2d = read_landmark(hm_3d_path)  #landmark 是以0为起点的
    landmark_2d[:, 1] = height - 1 - landmark_2d[:, 1]  # 坐标系原点转化为左下角
    tmp = np.zeros((landmark_2d.shape[0], 1))
    landmark_v = np.hstack((landmark_2d, tmp))
    landmark_body = np.array([[586,336,130],[562,369,150],[709,295,160],[727,262,150]])
    landmark_body[:,1] = 683-landmark_body[:,1]
    target_lmk_v = np.concatenate((landmark_v[:, 0:3], landmark_body[:, 0:3]))  # 三维landmark点

    # landmark embedding
    from fitting.landmarks import load_embedding
    from fitting.util import mesh_points_by_barycentric_coordinates
    frame_lmk_emb_path = './data/lmk_embedding_intraface_to_flame.pkl'
    lmk_face_idx, lmk_b_coords = load_embedding(frame_lmk_emb_path)
    #68 个 landmark
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
    from smpl_webuser.serialization import load_model
    model = load_model(frame_model_path)

    result = pre_result
    v = result['mesh_v']
    face = result['mesh_f']
    parms = result['parms']
    T = parms['trans']
    pose = parms['pose']
    betas = parms['betas']
    R = parms['global_rotate']
    Scale = parms['scale']
    model.betas[:] = betas[:]
    v_frame_init = model.r
    noeye_mesh, noeye_mesh_f, t_noeye, t_f_noeye, n_noeye, n_f_noeye = read_igl_obj(
        'D:\mproject/face-reconstruct\output/flame_blendshape/test_transfer/fast_transfer/Objec0001.obj')
    shape_model = shape_model

    # v_frame_align_to_image = (Scale * np.dot(R, v_frame_init.T)).T + T
    # v_selected_3d = mesh_points_by_barycentric_coordinates(v_frame_align_to_image, face, lmk_face_idx, lmk_b_coords)
    # source_face_lmkvtx = v_frame_align_to_image[face_select_lmk[0:17]]
    # use_lunkuo =1
    # if use_lunkuo:
    #     frame_landmark_idx = range(0,17)+range(17, 60) + range(61, 64) + range(65, 68)
    # else:
    #     frame_landmark_idx = range(17, 60) + range(61, 64) + range(65, 68)
    # if use_lunkuo:
    #     v_selected_merge = np.vstack([source_face_lmkvtx, v_selected_3d])
    # else:
    #     v_selected_merge = v_selected_3d
    # source_landmark = v_selected_merge

    index_i = 0
    #取出上一帧中的 形状，旋转，缩放，平移
    # faceshape = shapes[frame_t]
    # R = R_seq[frame_t]
    # s = s_seq[frame_t]
    # t = t_seq[frame_t]
    pre_R = R
    pre_s= Scale
    pre_t = T[0:2]
    s= pre_s
    t = pre_t
    pre_beta = np.zeros((1,shape_model.n_expresses))
    from frame_morphable_model import optexpressFromRST_t,optRSTfromShape_t
    featurePointxy = landmark_2d
    from triangle_raster import BBoxi_2d
    bbox2d = BBoxi_2d(featurePointxy)
    p1 = np.array([bbox2d.min[0],bbox2d.min[1]])
    p2 = np.array([bbox2d.max[0],bbox2d.max[1]])
    boundShape = np.linalg.norm(p1-p2)
    boundShape = 1. / boundShape
    faceshape = shape_model.compute_shape_from_blendshape(pre_beta)
    while index_i < 10:
        R, s, t = optRSTfromShape_t(faceshape,face, R, s, t, featurePointxy, lmk_face_idx, lmk_b_coords,face_select_lmk, pre_R, pre_s, pre_t)
        beta = optexpressFromRST_t(shape_model,face, R, s, t, featurePointxy, lmk_face_idx, lmk_b_coords,face_select_lmk,pre_beta, boundShape)
        faceshape = shape_model.compute_shape_from_blendshape(beta)
        index_i = index_i + 1
    v_frame_align_to_image = (s * np.dot(R, faceshape.T)).T + np.array([t[0],t[1],0])
    #write_simple_obj( v_frame_align_to_image, noeye_mesh_f,out_put_dir + object_name+'_expression' + '.obj')

    ori_image = ori_image[::-1,:,:]
    vertex_normal = get_vertex_normal(v_frame_align_to_image,noeye_mesh_f)
    v_color = sample_color_from_img(v_frame_align_to_image, vertex_normal, ori_image)
    frame_re_texture_map = 'D:\mproject/face-reconstruct/texpc\source_para/texture/frame_template_retex.obj'
    v_frame_aligned, f_frame_aligned, t_frame_aligned, t_f_frame_aligned, n_frame_aligned, n_f_frame_aligned = read_igl_obj(
        frame_re_texture_map)
    safe_mkdirs(out_put_dir)
    write_full_obj(v_frame_align_to_image, noeye_mesh_f, vertex_normal, noeye_mesh_f, np.array([]), np.array([]), v_color,out_put_dir+'/face_expression_texture.obj')
    pass

def generate_expression_simp(pre_result,vrn_object_dir,object_name,project_dir,out_put_dir,frame_model_path='./models/male_model.pkl'):
    # 先生成blendshape
    safe_mkdirs(out_put_dir)
    from smpl_webuser.serialization import load_model
    from  frame_morphable_model import FaceModel
    model = load_model(frame_model_path)

    result = pre_result
    v = result['mesh_v']
    face = result['mesh_f']
    parms = result['parms']
    T = parms['trans']
    pose = parms['pose']
    betas = parms['betas']
    R = parms['global_rotate']
    Scale = parms['scale']
    model.betas[:] = betas[:]
    v_frame_init = model.r
    faceshape = v_frame_init[0:3931]  # 不包括眼珠的顶点
    noeye_mesh, noeye_mesh_f, t_noeye, t_f_noeye, n_noeye, n_f_noeye = read_igl_obj(
        'D:\mproject/face-reconstruct\output/flame_blendshape/test_transfer/fast_transfer/Objec0001.obj')
    no_eye_face = noeye_mesh_f
    face_model = FaceModel(faceshape, no_eye_face)
    face_model.compute_Laplacian()
    face_model.generate_f_list()
    face_model.generate_blendshape_from_b0(faceshape)
    # 输出blendshape 到文件夹
    for i in range(0, 46):
        pass
        beta = np.zeros(46)
        beta[i] = 1.0
        faceshape = face_model.compute_shape_from_blendshape(beta)
        safe_mkdirs(out_put_dir + object_name + '/blendshape/')
        write_simple_obj(faceshape, noeye_mesh_f,
                         out_put_dir + object_name + '/blendshape/' + 'expression_' + str(i).zfill(3) + '.obj')
    from hello_world import render_to_image
    safe_mkdirs(out_put_dir + object_name + '/blendshape/' + 'render/')
    if 1:
        render_to_image(input_mesh_dir=out_put_dir +object_name+'/blendshape/',
                        out_put_dir=out_put_dir +object_name+'/blendshape/'+'render/')


def generate_expression(pre_result,vrn_object_dir,object_name,project_dir,out_put_dir,frame_model_path='./models/male_model.pkl'):
    #先生成blendshape
    safe_mkdirs(out_put_dir)
    from smpl_webuser.serialization import load_model
    from  frame_morphable_model import FaceModel
    model = load_model(frame_model_path)

    result = pre_result
    v = result['mesh_v']
    face = result['mesh_f']
    parms = result['parms']
    T = parms['trans']
    pose = parms['pose']
    betas = parms['betas']
    R = parms['global_rotate']
    Scale = parms['scale']
    model.betas[:] = betas[:]
    v_frame_init = model.r
    faceshape =v_frame_init[0:3931] # 不包括眼珠的顶点
    noeye_mesh, noeye_mesh_f, t_noeye, t_f_noeye, n_noeye, n_f_noeye = read_igl_obj(
        'D:\mproject/face-reconstruct\output/flame_blendshape/test_transfer/fast_transfer/Objec0001.obj')
    no_eye_face = noeye_mesh_f
    face_model = FaceModel(faceshape,no_eye_face)
    face_model.compute_Laplacian()
    face_model.generate_f_list()
    face_model.generate_blendshape_from_b0(faceshape)
    #输出blendshape 到文件夹
    for i in range(0,46):
        pass
        beta = np.zeros(46)
        beta[i] = 1.0
        faceshape = face_model.compute_shape_from_blendshape(beta)
        safe_mkdirs(out_put_dir +object_name+'/blendshape/')
        write_simple_obj(faceshape, noeye_mesh_f, out_put_dir +object_name+'/blendshape/'+ 'expression_' +str(i).zfill(3)+ '.obj')
    from hello_world import  render_to_image
    safe_mkdirs(out_put_dir +object_name+'/blendshape/'+'render/')
    if 0:
        render_to_image(input_mesh_dir=out_put_dir +object_name+'/blendshape/',
                        out_put_dir=out_put_dir +object_name+'/blendshape/'+'render/')
    #为每一帧求解blendshape 系数
    b = FileFilt()
    b.FindFile(dirr=project_dir)
    for k in b.fileList:
        if k == '':
            continue
            #       print k.split("/")[-1]
        # print k.split("/")
        filename_split = k.split("/")[-1].split(".")
        #        print filename_split
        if len(filename_split) > 1:
            print str(filename_split[-2])
            file_name = str(filename_split[-2])
            fomat_name = str(filename_split[-1])
        else:
            file_name = str(filename_split[0])
            fomat_name = ''
        obj_name= file_name
        genetate_expression_single(shape_model=face_model,
                                   preresult=pre_result,
                                   vrn_object_dir=project_dir + obj_name + '/',
                                   object_name=obj_name,
                                   project_dir=project_dir,
                                   out_put_dir=project_dir + obj_name + '/generate_face_expression2d/',
                                   use_3d_landmark=False
                                   )

    pass
if __name__ == '__main__':
    # generate_face(vrn_object_dir ='E:/workspace/vrn_data/bgBlue/A13010436665609/',
    #               object_name='A13010436665609',
    #               out_put_dir ='E:/workspace/vrn_data/bgBlue/A13010436665609/generate_face/')

    input_mesh_path ='E:/workspace/dataset/hairstyles/2d_hair/result/A13010436665609/builded_hair/frame_aligned_to_image.obj'
    input_dir = 'E:/workspace/dataset/hairstyles/2d_hair/result/A13010436665609/builded_hair/'
    seg_image_path = 'E:/workspace/dataset/hairstyles/2d_hair/result/A13010436665609/builded_hair/A13010436665609.png'
    ori_image_path = 'E:/workspace/dataset/hairstyles/2d_hair/A13010436665609.jpg'
    v_frame_aligned, f_frame_aligned, t_frame_aligned, t_f_frame_aligned, n_frame_aligned, n_f_frame_aligned = read_igl_obj(input_mesh_path)
    import cv2
    import os
    from fitting.util import smooth_seg_image,get_binaray_img_boundary,test_frame_mesh_landmark,read_int,write_landmark_to_obj
    #seg_img = cv2.imread(seg_image_path, cv2.IMREAD_COLOR)
    ori_img = cv2.imread(ori_image_path, cv2.IMREAD_COLOR)
    #smooth_seg_image(seg_image_path, input_dir+'A13010436665609_binary.png')
    seg_img_binary = cv2.imread(input_dir+'A13010436665609_face_2.png', cv2.IMREAD_GRAYSCALE)
    seg_img_binary = seg_img_binary>1
    if 0:
        face_img = np.zeros( (seg_img_binary.shape[0],seg_img_binary.shape[1],3),np.uint8)
        for j in range(0,seg_img_binary.shape[0]):
            for i in range(0,seg_img_binary.shape[1]):
                if seg_img_binary[j,i]:
                    face_img[j,i] = ori_img[j,i]

        #cv2.imwrite(input_dir+'A13010436665609_face.png',face_img)
        render_img, z_buffer_img = get_z_buffer(v_frame_aligned, f_frame_aligned, ori_img)
        for j in range(0,z_buffer_img.shape[0]):
            for i in range(0,z_buffer_img.shape[1]):
                if not seg_img_binary[j,i] or z_buffer_img[j,i] <3:
                    z_buffer_img[j,i] = 0
                    pass
        boundary_pixel = get_binaray_img_boundary(z_buffer_img)
        for j in range(0,z_buffer_img.shape[0]):
            for i in range(0,z_buffer_img.shape[1]):
                if [i,j] in boundary_pixel:
                    #z_buffer_img[j,i] = 255
                    pass
        boundary_pixel = np.array(boundary_pixel)
        #cv2.imwrite(input_dir + 'A13010436665609_face_zbuffer_boundary.png', z_buffer_img)
    if 0:
        Target = {'z0': z_buffer_img}
        mat_save(Target, input_dir + 'A13010436665609_face_zbuffer.mat')
        #shape_from_shading_detail(z_buffer_img, ori_img, seg_img_binary,boundary_pixel)
    #test_pncc()
    #test_scale_to_pncc()
    #test_scale_to_ori()

    pncc_landmark = np.array([21609,21806,22051,22221,22387,23072,23237,23446,23738,24060,8295,8304,8311,8316,6393,7171,8329,9615,10775,
                              1700,3755,4658,6476,4676,3644,10079,11103,12521,14331,12667,11635,5133,5897,7312,8341,9369,10791,11705,10173,
                              9012,8370,7729,6697,7580,8351,9121,7721,8362,9004
                              ])
    input_mesh_path = 'E:\workspace\dataset\hairstyles/templat_xyz.obj'
    # v_frame_aligned, f_frame_aligned, t_frame_aligned, t_f_frame_aligned, n_frame_aligned, n_f_frame_aligned = read_igl_obj(
    #     input_mesh_path)
    #fitting_frame_to_mesh_by3d_landmark(v_frame_aligned, f_frame_aligned, pncc_landmark, 'E:\workspace\dataset\hairstyles/test_pncc/')
    #test_landmark()
    input_model_dir ='E:/workspace/vrn_data/bgBlue/A13010436665609/generate_face/'
    #shape_from_shading(input_model_dir,ori_img, seg_img_binary,input_model_dir)
    if 0:
        extract_detail(subdived_mesh_path=input_model_dir+'/'+'plane_subdiv2.obj',ori_img=ori_img,seg_img=seg_img_binary,
                       output_dir='E:/workspace/vrn_data/bgBlue/A13010436665609/generate_face/test_plane/')
    if 0:
        input_mesh_path = 'D:\mproject/face-reconstruct/texpc/new_registration/rnd_head.obj'
        input_mesh_path = 'D:\mproject/face-reconstruct/texpc/new_registration/rnd_head_normalized.obj'
        out_put_path ='D:\mproject/face-reconstruct/texpc/new_registration/rnd_head_normalized.obj'
        v_frame_aligned, f_frame_aligned, t_frame_aligned, t_f_frame_aligned, n_frame_aligned, n_f_frame_aligned = read_igl_obj(
            input_mesh_path)
        landmark_idx = read_int('D:\mproject/face-reconstruct/texpc/new_registration/template_face_landmark_fix.txt')
        landmark_idx = landmark_idx[:,0]
        vcolor =readVertexColor(input_mesh_path)
        landmark3d = v_frame_aligned[landmark_idx]
        write_landmark_to_obj('D:\mproject/face-reconstruct/texpc/new_registration/rnd_landmark.obj',landmark3d,10)
        #v_frame_aligned/=100000
        #write_full_obj(v_frame_aligned,f_frame_aligned,n_frame_aligned, n_f_frame_aligned,t_frame_aligned, t_f_frame_aligned,vcolor,out_put_path)
        # fitting_frame_to_mesh_by3d_landmark(v_frame_aligned, f_frame_aligned, pncc_landmark,
        #                                     'E:\workspace\dataset\hairstyles/test_pncc/')

    if 0:
        extract_detailnew(input_obj='E:/workspace/vrn_data/bgBlue/A13010436665609/generate_face/triangle_mesh_1_smooth2.obj', output_dir='E:/workspace/vrn_data/bgBlue/A13010436665609/generate_face/test_smooth/')
    if 0:
        extract_detailnew(
            input_obj='E:/workspace/vrn_data/bgBlue/A13010436665609/generate_face/6_with_color.obj',
            output_dir='E:/workspace/vrn_data/bgBlue/A13010436665609/generate_face/test_plane_mesh_color/')
    if 0:
        generate_mesh_with_color()
    if 0:
        extract_detailnew(
            input_obj='E:/workspace/vrn_data/bgBlue/A13010436665609/generate_face/A1301074625120A_with_color.obj',
            output_dir='E:/workspace/vrn_data/bgBlue/A13010436665609/generate_face/test_plane_mesh_color/',
            objname='A1301074625120A')
    if 0:
        obj_name = 'collage-young-stylish-man-expressions-white-background-35379705_05' #useful-faces-picture-id108686357_09
        obj_name = 'useful-faces-picture-id108686357_09'
        pre_result = generate_face(frame_model_path = './models/generic_model.pkl',vrn_object_dir ='E:/workspace/vrn_data/expression_hair/'+obj_name+'/',
                       object_name=obj_name,
                       out_put_dir ='E:/workspace/vrn_data/expression_hair/'+obj_name+'/generate_face/',use_3d_landmark=False)
        objnames = ['useful-faces-picture-id108686357_01','useful-faces-picture-id108686357_02','useful-faces-picture-id108686357_03','useful-faces-picture-id108686357_04',
                    'useful-faces-picture-id108686357_05','useful-faces-picture-id108686357_06','useful-faces-picture-id108686357_07','useful-faces-picture-id108686357_08','useful-faces-picture-id108686357_10',
                    'useful-faces-picture-id108686357_11','useful-faces-picture-id108686357_12','useful-faces-picture-id108686357_13'
                    ]
        objnames =['useful-faces-picture-id108686357_14','useful-faces-picture-id108686357_15','useful-faces-picture-id108686357_16','useful-faces-picture-id108686357_17','useful-faces-picture-id108686357_18']
        objnames =['collage-young-stylish-man-expressions-white-background-35379705_01','collage-young-stylish-man-expressions-white-background-35379705_02',
                   'collage-young-stylish-man-expressions-white-background-35379705_03','collage-young-stylish-man-expressions-white-background-35379705_04','collage-young-stylish-man-expressions-white-background-35379705_06'
                   ]
        objnames = ['useful-faces-picture-id108686357_01']
        for i in range(0,len(objnames)):
            objectname =objnames[i]
            generate_face(frame_model_path='./models/generic_model.pkl',
                                       vrn_object_dir='E:/workspace/vrn_data/expression_hair/'+objectname,
                                       object_name=objectname,
                                       out_put_dir='E:/workspace/vrn_data/expression_hair/'+objectname+'/generate_face/',
                                       use_3d_landmark=False,pre_result=pre_result)
    if 0:
        pass
        project_dir = 'E:/workspace/vrn_data/wang_crop/select/'
        project_dir = 'E:/workspace/vrn_data/test_large_pose_2/'
        project_dir = 'E:/workspace/vrn_data/test_large_pose_woman/'

        obj_name = '000'
        pre_result = None
        if 0:
            pre_result = generate_face(frame_model_path='./models/female_model.pkl',
                                       vrn_object_dir=project_dir + obj_name + '/',
                                       object_name=obj_name,
                                       project_dir=project_dir,
                                       out_put_dir=project_dir + obj_name + '/generate_face/',
                                       use_3d_landmark=False)
        reference_obj = obj_name
        b = FileFilt()
        b.FindFile(dirr=project_dir)

        for k in b.fileList:
            if k == '':
                continue
                #       print k.split("/")[-1]
            # print k.split("/")
            filename_split = k.split("/")[-1].split(".")
            #        print filename_split
            if len(filename_split) > 1:
                print str(filename_split[-2])
                file_name = str(filename_split[-2])
                fomat_name = str(filename_split[-1])
            else:
                file_name = str(filename_split[0])
                fomat_name = ''
            if reference_obj == file_name:
                continue
            obj_name = file_name
            # if os.path.exists(project_dir+obj_name+'/generate_face/'+'face_with_texture.obj'):
            #     continue
            if obj_name == '08':
                continue
            pre_result = generate_face(frame_model_path='./models/female_model.pkl',
                                       vrn_object_dir=project_dir + obj_name + '/',
                                       object_name=obj_name,
                                       project_dir=project_dir,
                                       out_put_dir=project_dir + obj_name + '/generate_face/', use_3d_landmark=False,pre_result=pre_result)
    if 0:
        pass
        project_dir = 'E:/workspace/vrn_data/wang_crop/select/'
        obj_name = '000'
        input_model_dir = project_dir + obj_name + '/generate_face/'
        if 1 and os.path.exists(input_model_dir + '/face_result.pkl'):
            pre_result = load_binary_pickle(filepath=input_model_dir + '/face_result.pkl')
        else:
            pre_result = generate_face(frame_model_path='./models/male_model.pkl',
                                       vrn_object_dir=project_dir + obj_name + '/',
                                       object_name=obj_name,
                                       project_dir=project_dir,
                                       out_put_dir=project_dir + obj_name + '/generate_face/',
                                       use_3d_landmark=False)
        generate_expression(pre_result=pre_result, vrn_object_dir=project_dir + obj_name + '/', object_name= obj_name, project_dir=project_dir, out_put_dir=project_dir,
                            frame_model_path='./models/male_model.pkl')

    if 1:
        pass


    if 0: #generate shape for natural expression
        project_dir = 'E:\workspace/vrn_data\hairstyle/man/'
        project_dir = 'E:\workspace/vrn_data\paper_select1/man/'
        project_dir = 'E:\workspace/vrn_data\paper_compare/neatrual_1/man/'
        project_dir = 'D:\huayunhe/facewarehouse_new/FaceWarehouse_neutral_img_b/male/'

        b = FileFilt()
        b.FindFile(dirr=project_dir)

        for k in b.fileList:
            if k == '':
                continue
                #       print k.split("/")[-1]
            # print k.split("/")
            filename_split = k.split("/")[-1].split(".")
            #        print filename_split
            if len(filename_split) > 1:
                print str(filename_split[-2])
                file_name = str(filename_split[-2])
                fomat_name =  str(filename_split[-1])
            else:
                file_name = str(filename_split[0])
                fomat_name =''
            obj_name = file_name
            # if os.path.exists(project_dir+obj_name+'/generate_face/'+'face_with_texture.obj'):
            #     continue

            pre_result = generate_face(frame_model_path = './models/male_model.pkl',vrn_object_dir =project_dir+obj_name+'/',
                           object_name=obj_name,
                            project_dir=project_dir,
                           out_put_dir =project_dir+obj_name+'/generate_face/',use_3d_landmark=False)
        project_dir = 'E:\workspace/vrn_data\hairstyle/women/'
        project_dir = 'E:\workspace/vrn_data\paper_select1/women/'
        project_dir = 'E:\workspace/vrn_data\paper_compare/neatrual_1/women/'
        project_dir = 'D:\huayunhe/facewarehouse_new/FaceWarehouse_neutral_img_b/female/'
        b = FileFilt()
        b.FindFile(dirr=project_dir)

        for k in b.fileList:
            if k == '':
                continue
                #       print k.split("/")[-1]
            # print k.split("/")
            filename_split = k.split("/")[-1].split(".")
            #        print filename_split
            if len(filename_split) > 1:
                print str(filename_split[-2])
                file_name = str(filename_split[-2])
                fomat_name =  str(filename_split[-1])
            else:
                file_name = str(filename_split[0])
                fomat_name =''
            obj_name = file_name
            # if os.path.exists(project_dir+obj_name+'/generate_face/'+'face_with_texture.obj'):
            #     continue
            pre_result = generate_face(frame_model_path = './models/female_model.pkl',vrn_object_dir =project_dir+obj_name+'/',
                           object_name=obj_name,
                           project_dir=project_dir,
                           out_put_dir =project_dir+obj_name+'/generate_face/',use_3d_landmark=False)

    if 0: # generate blend shape for natural shape
        project_dir = 'E:\workspace/vrn_data\paper_compare/neatrual_1/man/'

        b = FileFilt()
        b.FindFile(dirr=project_dir)

        for k in b.fileList:
            if k == '':
                continue
                #       print k.split("/")[-1]
            # print k.split("/")
            filename_split = k.split("/")[-1].split(".")
            #        print filename_split
            if len(filename_split) > 1:
                print str(filename_split[-2])
                file_name = str(filename_split[-2])
                fomat_name = str(filename_split[-1])
            else:
                file_name = str(filename_split[0])
                fomat_name = ''
            obj_name = file_name
            # if os.path.exists(project_dir+obj_name+'/generate_face/'+'face_with_texture.obj'):
            #     continue
            input_model_dir = project_dir + obj_name + '/generate_face/'
            if  os.path.exists(input_model_dir + '/face_result.pkl'):
                pre_result = load_binary_pickle(filepath=input_model_dir + '/face_result.pkl')
            generate_expression_simp(pre_result=pre_result, vrn_object_dir=project_dir + obj_name + '/', object_name= obj_name, project_dir=project_dir, out_put_dir=project_dir,
                                frame_model_path='./models/male_model.pkl')
        project_dir = 'E:\workspace/vrn_data\hairstyle/women/'
        project_dir = 'E:\workspace/vrn_data\paper_select1/women/'
        project_dir = 'E:\workspace/vrn_data\paper_compare/neatrual_1/women/'
        b = FileFilt()
        b.FindFile(dirr=project_dir)

        for k in b.fileList:
            if k == '':
                continue
                #       print k.split("/")[-1]
            # print k.split("/")
            filename_split = k.split("/")[-1].split(".")
            #        print filename_split
            if len(filename_split) > 1:
                print str(filename_split[-2])
                file_name = str(filename_split[-2])
                fomat_name = str(filename_split[-1])
            else:
                file_name = str(filename_split[0])
                fomat_name = ''
            obj_name = file_name
            # if os.path.exists(project_dir+obj_name+'/generate_face/'+'face_with_texture.obj'):
            #     continue
            input_model_dir = project_dir + obj_name + '/generate_face/'
            if  os.path.exists(input_model_dir + '/face_result.pkl'):
                pre_result = load_binary_pickle(filepath=input_model_dir + '/face_result.pkl')
            generate_expression_simp(pre_result=pre_result, vrn_object_dir=project_dir + obj_name + '/', object_name= obj_name, project_dir=project_dir, out_put_dir=project_dir,
                                frame_model_path='./models/female_model.pkl')

        pass

    if 0:
        landmark3d = read_landmark('E:\workspace/vrn_data/pointlist.txt','	')

        #write_landmark_to_obj('E:\workspace/vrn_data/pointlist.obj',landmark3d,10)
        igl.writeOBJ('E:\workspace/vrn_data/' + '/pointlist_landmark' + '.obj', igl.eigen.MatrixXd(landmark3d.astype('float64')),
                    igl.eigen.MatrixXi(landmark_face.astype('intc')))
    if 0:
        landmark_v, landmark_f, t, t_f, n, n_f = read_igl_obj('E:\workspace/vrn_data/' + '/pointlist_landmark' + '.obj')
        subdived_mesh, subdived_mesh_f, t_frame_aligned, t_f_frame_aligned, n_frame_aligned, n_f_frame_aligned = read_igl_obj(
            'E:\workspace/vrn_data/shape_0.obj')
        if 0:
            frame_landmark_idx = range(0,17)+range(17, 60) + range(61, 64) + range(65, 68)
        else:
            frame_landmark_idx = range(17, 60) + range(61, 64) + range(65, 68)
        target_lmk_3d_face = landmark_v[frame_landmark_idx,:]
        from sklearn.neighbors import NearestNeighbors

        neigh = NearestNeighbors(n_neighbors=1)
        neigh.fit(subdived_mesh)
        distances, indices = neigh.kneighbors(target_lmk_3d_face, return_distance=True)
        landmark_indices = indices[:,0]
        fitting_frame_to_mesh_by3d_landmark(subdived_mesh, subdived_mesh_f, landmark_indices,
                                            'E:\workspace/vrn_data/')
    if 0:
        #landmark_v, landmark_f, t, t_f, n, n_f = read_igl_obj('E:\workspace/vrn_data/' + '/pointlist_landmark' + '.obj')
        landmark3d = read_landmark('D:\mproject/face-reconstruct/texpc/new_registration/template_face_landmark_fix.txt', '\n')
        subdived_mesh, subdived_mesh_f, t_frame_aligned, t_f_frame_aligned, n_frame_aligned, n_f_frame_aligned = read_igl_obj(
            'D:\mproject/face-reconstruct/texpc/new_registration/rnd_head_normalized.obj')
        if 0:
            frame_landmark_idx = range(0,17)+range(17, 60) + range(61, 64) + range(65, 68)
        else:
            frame_landmark_idx = range(17, 60) + range(61, 64) + range(65, 68)

        landmark_indices = landmark3d[frame_landmark_idx,:]
        landmark_indices = landmark_indices.astype(int)
        landmark_indices =landmark_indices[:,0]
        fitting_frame_to_mesh_by3d_landmark(subdived_mesh, subdived_mesh_f, landmark_indices,
                                            'D:\mproject/face-reconstruct/texpc/new_registration/')


    if 1:
        from fitting.util import convert_img_2_mseh_new
        pass
        downsample_rate = 2.0
        project_object = 'orlandobloom-2af2e3c15a33ceece6d7ef0644e96154-1200x600'
        project_object='2014-Men-Hairstyles-2'
        project_object='chris-hemsworth-long-hair'
        project_object = '001_1'
        project_object = '15'
        #project_object = '002_1'
        #project_object = '004_1'
        project_dir = 'E:\workspace/vrn_data/paper_select1\man/'+project_object+'/'
        frame_re_texture_map = 'D:\mproject/face-reconstruct/texpc\source_para/texture/frame_template_retex.obj'
        first_run = True
        if 0:
            v_frame_aligned, f_frame_aligned, t_frame_aligned, t_f_frame_aligned, n_frame_aligned, n_f_frame_aligned = read_igl_obj(
                frame_re_texture_map)
            generate_mesh, generate_mesh_f, t_generate_mesh, t_f_generate_mesh, n_generate_mesh, n_f_generate_mesh = read_igl_obj(
                project_dir+'/generate_face/'+'face.obj')
            generate_mesh_n = get_vertex_normal(generate_mesh, generate_mesh_f)
            write_full_obj(generate_mesh,generate_mesh_f,generate_mesh_n,n_f_frame_aligned,t_frame_aligned,t_f_frame_aligned,np.array([]),
                           project_dir + '/generate_face/' +'face_with_texture' + '.obj')


        #为细分后的网格上色
        ori_image_path = project_dir + project_object+'.jpg'
        ori_img = cv2.imread(ori_image_path, cv2.IMREAD_COLOR)
        ori_img = cv2.resize(ori_img, (int(ori_img.shape[1]/downsample_rate), int(ori_img.shape[0]/downsample_rate)), interpolation=cv2.INTER_CUBIC)
        if 0:
            smooth_seg_image(project_dir+'face_seg.png', project_dir+'face_seg_smooth.png')
        seg_img_binary = cv2.imread(project_dir+'face_seg_smooth.png', cv2.IMREAD_GRAYSCALE)
        seg_img_binary = cv2.resize(seg_img_binary,(int(ori_img.shape[1] ), int(ori_img.shape[0])),interpolation=cv2.INTER_NEAREST)
        seg_img_binary = seg_img_binary>1
        if 0:
            subdived_mesh, subdived_mesh_f, t_frame_aligned, t_f_frame_aligned, n_frame_aligned, n_f_frame_aligned = read_igl_obj(
                project_dir+'generate_face/'+'face_subdiv_refined.obj')
            subdived_mesh = subdived_mesh / downsample_rate
            subdived_mesh_n = get_vertex_normal(subdived_mesh, subdived_mesh_f)
            vcolor = sample_color_from_img(subdived_mesh, subdived_mesh_n, ori_img[::-1,:,:], seg_img_binary[::-1,:])

            write_full_obj(subdived_mesh,subdived_mesh_f,n_frame_aligned,n_f_frame_aligned,t_frame_aligned,t_f_frame_aligned,vcolor,project_dir + '/generate_face/' +'subdiv_with_color' + '.obj')
        #生成深度图网格
        if 0:
            subdived_mesh, subdived_mesh_f, t_frame_aligned, t_f_frame_aligned, n_frame_aligned, n_f_frame_aligned = read_igl_obj(
                project_dir+'generate_face/'+'subdiv_with_color.obj')
            vcolor = readVertexColor(project_dir + 'generate_face/' + 'subdiv_with_color.obj')
            render_img, z_buffer_img = get_z_buffer(subdived_mesh, subdived_mesh_f, ori_img,output_render_img =project_dir+'render_img_0.png',vcolor=vcolor)
            new_vertex, new_face, new_color = convert_img_2_mseh_new(ori_img[:,:,:],seg_img_binary,z_buffer_img)
            new_color = new_color/255.0
            write_full_obj(new_vertex, new_face, np.array([]), np.array([]), np.array([]),
                           np.array([]),new_color,
                           project_dir + '/generate_face/' + 'zbuffer_with_color' + '.obj')
        if 0:
            for j in range(0,z_buffer_img.shape[0]):
                for i in range(0,z_buffer_img.shape[1]):
                    if not seg_img_binary[j,i] or z_buffer_img[j,i] <3:
                        z_buffer_img[j,i] = 0
                        pass
            boundary_pixel = get_binaray_img_boundary(z_buffer_img)
            for j in range(0,z_buffer_img.shape[0]):
                for i in range(0,z_buffer_img.shape[1]):
                    if [i,j] in boundary_pixel:
                        #z_buffer_img[j,i] = 255
                        pass
            boundary_pixel = np.array(boundary_pixel)
            cv2.imwrite(project_dir+'render_img.png', render_img[:, :, ::-1])
            cv2.imwrite(project_dir+'z_buffer_img.png', z_buffer_img)
            depth_img = np.zeros((z_buffer_img.shape[0], z_buffer_img.shape[1], 3))
            for j in range(0, z_buffer_img.shape[0]):
                for i in range(0, z_buffer_img.shape[1]):
                    depth_img[j, i, 0] = i / 2.0
                    depth_img[j, i, 1] = j / 2.0
                    depth_img[j, i, 2] = z_buffer_img[j, i]
        if 1:
            extract_detailnew(
                input_obj=project_dir+'generate_face/'+'zbuffer_with_color_refined_smooth.obj',
                output_dir=project_dir+'generate_face/',
                objname='zbuffer_with_color_refine')
        if 0:
            extract_detailnew(
                input_obj=project_dir+'generate_face/'+'zbuffer_with_color_refine_detail_mesh_1.obj',
                output_dir=project_dir+'generate_face/',
                objname='zbuffer_with_color_refine2_')
    if 0:
        subdived_mesh, subdived_mesh_f, t_frame_aligned, t_f_frame_aligned, n_frame_aligned, n_f_frame_aligned = read_igl_obj('E:\workspace\dataset\hairstyles/frame_aligned.obj')
        optim_v = frame_head_optimaization(subdived_mesh,subdived_mesh_f)
        write_full_obj(optim_v, subdived_mesh_f, n_frame_aligned, n_f_frame_aligned, np.array([]), np.array([]), np.array([]), 'E:\workspace\dataset\hairstyles/frame_aligned_refined.obj')

    if 0:
        from sklearn.neighbors import NearestNeighbors
        subdived_mesh, subdived_mesh_f, t_frame_aligned, t_f_frame_aligned, n_frame_aligned, n_f_frame_aligned = read_igl_obj('D:\mproject/face-reconstruct/texpc/new_registration/fitting_frame_face.obj')
        subdived_mesh1, subdived_mesh_f1, t_frame_aligned1, t_f_frame_aligned1, n_frame_aligned1, n_f_frame_aligned1 = read_igl_obj(
            'D:\mproject/face-reconstruct/texpc/new_registration/rnd_head_normalized.obj')
        from fitting.measure import mesh2mesh,distance2color
        neigh = NearestNeighbors(n_neighbors=1)
        neigh.fit(subdived_mesh1)
        print 'neigh.fit'
        distances, indices = neigh.kneighbors(subdived_mesh, return_distance=True)
        distance = mesh2mesh(subdived_mesh,subdived_mesh1[indices[:,0]])
        print np.min(distance),np.max(distance)
        vmax = 0.03
        color_3d = distance2color(dist = distance, vmin=0, vmax=vmax, cmap_name='jet')
        write_full_obj(subdived_mesh, subdived_mesh_f, n_frame_aligned, n_f_frame_aligned, np.array([]), np.array([]),
                       color_3d, 'D:\mproject/face-reconstruct/texpc/new_registration/fitting_frame_face_error'+str(vmax)+'.obj')
    if 0:
        subdived_mesh, subdived_mesh_f, t_frame_aligned, t_f_frame_aligned, n_frame_aligned, n_f_frame_aligned = read_igl_obj(
            'E:\workspace/vrn_data\paper_select1\man/004_1\generate_face/face_with_texture.obj')
        vtx_color = readVertexColor('E:\workspace/vrn_data\paper_select1\man/004_1\generate_face/face_with_texture.obj')
        from fitting.util import mesh_loop
        mesh_loop(subdived_mesh, subdived_mesh_f, n_frame_aligned, n_f_frame_aligned,t_frame_aligned, t_f_frame_aligned,vtx_color,
                  'E:\workspace/vrn_data\paper_select1\man/004_1\generate_face/face_with_test_subdiv.obj',2 )


    if 0:
        pass
        project_dir = 'E:\workspace/vrn_data\expression1/'
        obj_name = 'useful-faces-picture-id108686357_09'
        input_model_dir = project_dir + obj_name + '/generate_face/'
        if os.path.exists(input_model_dir + '/face_result.pkl'):
            pre_result = load_binary_pickle(filepath=input_model_dir + '/face_result.pkl')
        else:
            pre_result = generate_face(frame_model_path='./models/male_model.pkl',
                                       vrn_object_dir=project_dir + obj_name + '/',
                                       object_name=obj_name,
                                       project_dir=project_dir,
                                       out_put_dir=project_dir + obj_name + '/generate_face/',
                                       use_3d_landmark=False)
        generate_expression(preresult=pre_result, vrn_object_dir=project_dir + obj_name + '/', object_name= obj_name, project_dir=project_dir, out_put_dir=project_dir,
                            frame_model_path='./models/male_model.pkl')

    if 0:
        color_sum = extract_para_texture(input_texture_img='E:\workspace/vrn_data\paper_select1\man/004_1\generate_face/subdiv1_full.png',
                             input_texture_seg='E:\workspace/vrn_data\paper_select1\man/004_1\generate_face/seg2.png',
                             para_dir='D:\mproject/face-reconstruct/texpc\source_para/texture_full/')
        cv2.imwrite('E:\workspace/vrn_data\paper_select1\man/004_1\generate_face/color_sum.png',color_sum)

    if 0:
        from smpl_webuser.serialization import load_model
        from fitting.landmarks import load_embedding, landmark_error_3d

        model = load_model('./models/male_model.pkl')  # the loaded model object is a 'chumpy' object, check https://github.com/mattloper/chumpy for details
        # landmark embedding
        lmk_face_idx, lmk_b_coords = load_embedding('./data/lmk_embedding_intraface_to_flame.pkl')
        # 68 个 landmark
        face_select_lmk = np.array(
            [2210, 1963, 3486, 3382, 3385, 3389, 3392, 3396, 3400, 3599, 3594, 3587, 3581, 3578, 3757, 568, 728,
             3764, 3158, 335, 3705, 2178,
             673, 3863, 16, 2139, 3893,
             3553, 3561, 3501, 3564,
             2747, 2749, 3552, 1617, 1611,
             2428, 2383, 2493, 2488, 2292, 2335,
             1337, 1342, 1034, 1154, 959, 880,
             2712, 2850, 2811, 3543, 1694, 1735, 1576, 1770, 1801, 3511, 2904, 2878,
             2715, 2852, 3531, 1737, 1579, 1793, 3504, 2896
             ])
        from fitting.util import mesh_points_by_barycentric_coordinates
        v_frame_init=model.r
        v_selected_3d = mesh_points_by_barycentric_coordinates(v_frame_init, model.f, lmk_face_idx, lmk_b_coords)
        source_face_lmkvtx = v_frame_init[face_select_lmk[0:17]]
        a =[60,64]
        add_l =  v_frame_init[face_select_lmk[a]]
        use_lunkuo =1
        if use_lunkuo:
            frame_landmark_idx = range(0,17)+range(17, 60) + range(61, 64) + range(65, 68)

        else:
            frame_landmark_idx = range(17, 60) + range(61, 64) + range(65, 68)
        if use_lunkuo:
            v_selected_merge = np.vstack([source_face_lmkvtx, v_selected_3d,add_l])
        else:
            v_selected_merge = v_selected_3d
        landmark3d = v_selected_merge
        write_landmark_to_obj('E:\workspace\dataset\hairstyles/flame_male_landmark.obj', landmark3d, 1)

    def test_extract_detail_time():
        from fitting.util import convert_img_2_mseh_new
        from time import time
        timer_start = time()
        timer_end = time()
        project_object = '002_1'
        project_dir = 'E:\workspace/vrn_data/paper_select1\man/' + project_object + '/'
        downsample_rate =0.1

        subdived_mesh, subdived_mesh_f, t_frame_aligned, t_f_frame_aligned, n_frame_aligned, n_f_frame_aligned = read_igl_obj(
            project_dir + 'test_detail/' + 'face_subdiv_refined.obj')
        # 为细分后的网格上色
        ori_image_path = project_dir + project_object + '.jpg'
        ori_img = cv2.imread(ori_image_path, cv2.IMREAD_COLOR)
        ori_img = cv2.resize(ori_img,
                             (int(ori_img.shape[1] / downsample_rate), int(ori_img.shape[0] / downsample_rate)),
                             interpolation=cv2.INTER_CUBIC)
        seg_img_binary = cv2.imread(project_dir+'face_seg_smooth.png', cv2.IMREAD_GRAYSCALE)
        seg_img_binary = cv2.resize(seg_img_binary,(int(ori_img.shape[1] ), int(ori_img.shape[0])),interpolation=cv2.INTER_NEAREST)
        seg_img_binary = seg_img_binary>1


        subdived_mesh = subdived_mesh / downsample_rate
        subdived_mesh_n = get_vertex_normal(subdived_mesh, subdived_mesh_f)
        vcolor = sample_color_from_img(subdived_mesh, subdived_mesh_n, ori_img[::-1, :, :], seg_img_binary[::-1, :])

        write_full_obj(subdived_mesh, subdived_mesh_f, n_frame_aligned, n_f_frame_aligned, t_frame_aligned,
                       t_f_frame_aligned, vcolor, project_dir + '/test_detail/' + 'subdiv_with_color' + '.obj')
        subdived_mesh, subdived_mesh_f, t_frame_aligned, t_f_frame_aligned, n_frame_aligned, n_f_frame_aligned = read_igl_obj(
            project_dir + 'test_detail/' + 'subdiv_with_color.obj')
        vcolor = readVertexColor(project_dir + 'test_detail/' + 'subdiv_with_color.obj')
        render_img, z_buffer_img = get_z_buffer(subdived_mesh, subdived_mesh_f, ori_img,
                                                output_render_img=project_dir + 'render_img_0.png')
        new_vertex, new_face, new_color = convert_img_2_mseh_new(ori_img[:, :, :], seg_img_binary, z_buffer_img)
        new_color = new_color / 255.0
        write_full_obj(new_vertex, new_face, np.array([]), np.array([]), np.array([]),
                       np.array([]), new_color,
                       project_dir + '/test_detail/' + 'zbuffer_with_color_'+str(ori_img.shape[0])
                       +'_'+str(ori_img.shape[1]) + '.obj')
        pass

    def test_extract_time():
        from time import time

        project_object = '002_1'
        project_dir = 'E:\workspace/vrn_data/paper_select1\man/002_1/test_detail/test_mesh/'
        b = FileFilt()
        b.FindFile(dirr=project_dir)
        for k in b.fileList:
            if k == '':
                continue
                #       print k.split("/")[-1]
            # print k.split("/")
            filename_split = k.split("/")[-1].split(".")
            #        print filename_split
            if len(filename_split) > 1:
                print str(filename_split[-2])
                file_name = str(filename_split[-2])
                fomat_name = str(filename_split[-1])
            else:
                file_name = str(filename_split[0])
                fomat_name = ''
            obj_name = file_name
            timer_start = time()
            extract_detailnew(
                input_obj=project_dir + file_name + '.obj',
                output_dir=project_dir + 'result/',
                objname=file_name)
            timer_end = time()
            print file_name,timer_end-timer_start


    def test_extract_time():
        project_dir = 'E:\workspace/vrn_data/paper_select1\man/002_1/test_detail/test_mesh/'
        file_name ='zbuffer_with_color_180_141'
        from time import time
        for i  in range(0,20):
            timer_start = time()
            extract_detailnew(
                input_obj=project_dir + file_name + '.obj',
                output_dir=project_dir + 'result/',
                objname=file_name,
                dt1 = -0.2*i*i*i*i
            )
            timer_end = time()
            print file_name,str(-0.2*i*i*i*i),timer_end-timer_start
        pass
    #test_extract_detail_time()
    #test_extract_time()
    #test_extract_time()