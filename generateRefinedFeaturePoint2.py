# -- coding: utf-8 --

import sys
sys.path.insert(0, "D:/mprojects/libiglfull/libigl/python/build/x64/Release")
import pyigl as igl
from os.path import join
from fitting.util import safe_mkdir, mat_save,mat_load,IglMatrixTonpArray,FileFilt,write_simple_obj,k_main_dir_sklearn
from fitting.util import readImage,write_image_and_featurepoint,read_landmark, \
    cast2d_to3d_trimesh,scaleToOriCoodi_bottomleft,sym_plane,corr_point,\
    sym_point,write_landmark_to_obj,write_full_obj,corr_landmark_tofit_data,detect_68_landmark_dlib
import numpy as np
import  math
projectdir = 'L:/yuanqing/imgs/imgs/vrn_result/niutou/'
objeectdir=''
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
# sympoint of 68 point
point_sym = { 0:16,1:15,2:14,3:13,4:12,5:11,6:10,7:9,8:8,9:7,10:6,11:5,12:4,
             13:3,14:2,15:1,16:0,17:26,18:25,19:24,20:23,21:22,22:21,23:20,
             24:19,25:18,26:17,27:27,28:28,29:29,30:30,31:35,32:34,33:33,34:32,35:31,
             36:45,37:44,38:43,39:42,40:47,41:46,42:39,43:38,44:37,45:36,46:41,47:40,
             48:54,49:53,50:52,51:51,52:50,53:49,54:48,55:59,56:58,57:57,58:56,59:55,
             60:64,61:63,62:62,63:61,64:60,65:67,66:66,67:65}
'''
online_path = 'L:/yuanqing/imgs/imgs/vrn_result/niutou/online/'
a = FileFilt()
a.FindFile(dirr=online_path)
for k in a.fileList:
    if k == '':
        continue
    print k.split("/")[-1]
    filename_split = k.split("/")[-1].split(".")
    file_name = []
    if len(filename_split) > 1:
        file_name = str(filename_split[-2])
    v = igl.eigen.MatrixXd()
    f = igl.eigen.MatrixXi()

    igl.readOBJ(k, v, f)
    V = np.array(v)
    if V.shape[1]>3:
        V=V[:,0:3]
    V[:,1] = 192-V[:,1]  #以左下角为原点
    igl.writeOBJ(online_path+file_name+'_flip.obj', igl.eigen.MatrixXd(V.astype('float64')), f)
'''

b = FileFilt()
targetdir = projectdir+objeectdir
b.FindFile(dirr=targetdir)
print(b.counter)
target = 9
count = 0
for k in b.fileList:
    if k == '':
        continue
    if count != target:
        count+=1
        continue
    print k

    print k.split("/")[-1]
    #    print k.split("/")
    filename_split = k.split("/")[-1].split(".")
    file_name = []
    if len(filename_split) > 1:
        #       print str(filename_split[-2])
        file_name = str(filename_split[-2])
    objdir = targetdir + file_name
    result = mat_load(objdir + '/result/' + file_name)
    BB = result['BB']
    print BB
    BB = result['BB'][0]
    print BB
    result = result['result']
    V= result['vertices'][0,0]
    F = result['faces'][0, 0]
    vtx_color = result['FaceVertexCData'][0, 0]
    vtx_color = vtx_color/255.0
    F = F-1
    V[:,1] = 192-1-V[:,1]  #以左下角为原点
    V_igl = igl.eigen.MatrixXd(V.astype('float64'))
    F_igl = igl.eigen.MatrixXi(F.astype('intc'))


    ori_image = readImage(objdir+'/'+file_name+'.jpg')
    height,width,dim = ori_image.shape
    landmark_2d = read_landmark(objdir+'/2d/'+file_name+'.txt')
    write_image_and_featurepoint(ori_image,landmark_2d,objdir+'/'+'ori_feature_2d'+'.jpg')
    landmark_2d[:,1] = height-1 -landmark_2d[:,1] #坐标系原点转化为左下角
    use_dlib =0
    if use_dlib: #use dlib
        landmark_2d = detect_68_landmark_dlib(ori_image)
        write_image_and_featurepoint(ori_image, landmark_2d, objdir + '/' + 'dlib_feature_2d' + '.jpg')
        landmark_2d[:, 1] = height - landmark_2d[:, 1]

    expand_landmark = np.zeros((landmark_2d.shape[0],3))
    expand_landmark[:,0:2] = landmark_2d[:,0:2]
    expand_landmark[:, 2] = 192
    BB[1] = height-1 -BB[1]-BB[2]  #坐标系原点转化为左下角
    print 'left bottom BB ',BB

# 可能需要去vrn 模型中的一些错误数据
#
    VertexNormal = igl.eigen.MatrixXd()
    write_full_obj(V, F, np.array([]), np.array([]), np.array([]), np.array([]), vtx_color,
                   objdir + '/ori_color_crop' + '.obj', )
    import os
    if not os.path.exists(objdir + '/Target_corr' + '.obj'):

        igl.per_vertex_normals(V_igl, F_igl, igl.PER_VERTEX_NORMALS_WEIGHTING_TYPE_AREA, VertexNormal)
        # 1. 将裁减图转化到原图坐标，并设置为左下角为原点
        write_full_obj(V, F, np.array([]), np.array([]), np.array([]), np.array([]), vtx_color,
                       objdir + '/ori_color_crop' + '.obj', )
        target_V = scaleToOriCoodi_bottomleft(V, BB, 192)
        write_full_obj(target_V, F, np.array([]), np.array([]), np.array([]), np.array([]), vtx_color,
                       objdir + '/ori_color' + '.obj', )
        igl.writeOBJ(objdir + '/Target_corr' + '.obj', igl.eigen.MatrixXd(target_V),
                 igl.eigen.MatrixXi(F_igl), igl.eigen.MatrixXd(VertexNormal),
                 igl.eigen.MatrixXi(F_igl), igl.eigen.MatrixXd(), igl.eigen.MatrixXi())
 #       break
    else:
        n_f = igl.eigen.MatrixXi()
        t_v = igl.eigen.MatrixXd()
        t_f = igl.eigen.MatrixXi()
        igl.readOBJ(objdir + '/Target_corr' + '.obj', V_igl,t_v,VertexNormal,
                    F_igl,t_f,n_f)
        target_V = np.array(V_igl)
        if target_V.shape[1]>3:
            target_V = target_V[:,0:3]


# 2. 提取对称面
    mesh_center ,main_dir,eigenvalues =k_main_dir_sklearn(target_V,3)
    up_dir = -main_dir[0]
    vertical_normal = -main_dir[1]
    front_dir = main_dir[2]
    up_dir =up_dir/ np.linalg.norm(up_dir)
    vertical_normal =vertical_normal/ np.linalg.norm(vertical_normal)
    front_dir = front_dir / np.linalg.norm(front_dir)
    symplane_vertex, symplane_f = sym_plane(front_dir, up_dir, vertical_normal, mesh_center, 100)
    igl.writeOBJ(objdir + '/01_symplane_corr' + '.obj', igl.eigen.MatrixXd(symplane_vertex.astype('float64')),
                igl.eigen.MatrixXi(symplane_f.astype('intc')))

# 3 .通过二维投影的方式求三维点
    from time import time
    timer_start = time()
    landmark_cast = []
#    landmark_2d[:,0]+=1
#    landmark_2d[:,1] -= 1
    index_triangles, index_ray, result_locations = cast2d_to3d_trimesh(objdir + '/Target_corr' + '.obj',landmark_2d)

    bool_ray = np.zeros((landmark_2d.shape[0],1),np.int)
    for i_ray in index_ray:
        bool_ray[i_ray] = 1
    landmark_v = np.zeros((landmark_2d.shape[0],3),np.float64)
    for i in range(0,landmark_v.shape[0]):
        landmark_v[i,:] = [landmark_2d[i,0],landmark_2d[i,1],100]
    for i in range(0,result_locations.shape[0]):
        landmark_idx = index_ray[i]
        landmark_v[landmark_idx,:] = result_locations[i,:]

#    print index_triangles,index_ray,result_locations
    write_landmark_to_obj(objdir + '/feature_point_mesh_bias' + '.obj',result_locations)
    timer_end = time()
    print "ray cast %f sec\n" % (timer_end - timer_start)

# 4  .把中间点修正到对称面


    for i  in [27,28,29,30,33,51,62,66,57,8]:
        landmark_v[i,:] =corr_point(front_dir,up_dir,vertical_normal,mesh_center,landmark_v[i,:])
    igl.writeOBJ(objdir + '/01_corr_landmark_step1' + '.obj', igl.eigen.MatrixXd(landmark_v.astype('float64')),
                igl.eigen.MatrixXi(landmark_face.astype('intc')))
    for i in [27, 28, 29, 30, 33, 51, 62, 66, 57, 8]:
        landmark_v[i, :],buf_vertex = corr_landmark_tofit_data(front_dir, up_dir, vertical_normal, mesh_center, target_V,landmark_v[i, :])
    igl.writeOBJ(objdir + '/01_corr_landmark_step2' + '.obj', igl.eigen.MatrixXd(landmark_v.astype('float64')),
                igl.eigen.MatrixXi(landmark_face.astype('intc')))
    igl.writeOBJ(objdir + '/01_corr_landmark_step2_buf' + '.obj', igl.eigen.MatrixXd(buf_vertex.astype('float64')),
                 igl.eigen.MatrixXi())
    if vertical_normal.dot(np.array([0,0,-1])) > math.cos(math.radians(75)):
        print 'vertical_normal.dot(np.array([0,0,-1]))',vertical_normal.dot(np.array([0, 0, -1]))
        for i in [9,10,11,12,13,14,15,16,22,23,24,25,26,34,35,42,43,44,45,46,47,52,53,54,55,56,63,64,65]:
            sym_idx = point_sym[i]
            landmark_v[sym_idx, :] =sym_point(front_dir,up_dir,vertical_normal,mesh_center, landmark_v[i,:])
    elif vertical_normal.dot(np.array([0,0,1])) > math.cos(math.radians(75)):
        print 'vertical_normal.dot(np.array([0,0,1]))',vertical_normal.dot(np.array([0, 0, -1]))
        for i in [0,1,2,3,4,5,6,7,17,18,19,20,21,31,32,36,37,38,39,40,41,48,49,50,60,61,67,58,59]:
            sym_idx = point_sym[i]
            landmark_v[sym_idx, :] =sym_point(front_dir,up_dir,vertical_normal,mesh_center, landmark_v[i,:])
    igl.writeOBJ(objdir + '/01_corr_landmark_step3' + '.obj', igl.eigen.MatrixXd(landmark_v.astype('float64')),
                igl.eigen.MatrixXi(landmark_face.astype('intc')))
    landmark_v[:, 1] = height-1 - landmark_v[:, 1]

    if use_dlib: #use dlib
        write_image_and_featurepoint(ori_image, landmark_v[:, 0:2], objdir + '/' + 'corr_dlib_feature_2d' + '.jpg')
    else:
        write_image_and_featurepoint(ori_image, landmark_v[:, 0:2], objdir + '/' + 'corr_feature_2d' + '.jpg')

    if count  == target:
        break
 #   break