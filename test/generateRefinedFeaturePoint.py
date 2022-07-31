# -- coding: utf-8 --

import sys
sys.path.insert(0, "D:/mprojects/libiglfull/libigl/python/build/x64/Release")
import pyigl as igl
from os.path import join
from fitting.util import safe_mkdir, mat_save,mat_load,IglMatrixTonpArray,FileFilt,write_simple_obj,k_main_dir_sklearn
from fitting.util import readImage,write_image_and_featurepoint,read_landmark,\
    cast2d_to3d,scaleToOriCoodi_bottomleft,sym_plane,corr_point,sym_point,convertObj2Mat,convertMat2obj

import numpy as np
import random

projectdir = 'L:/yuanqing/imgs/imgs/vrn_result/niutou/'
objeectdir=''

import scipy
import scipy.cluster.hierarchy as sch
from scipy.cluster.vq import vq, kmeans, whiten
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pylab as plt

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

point_sym = { 0:16,1:15,2:14,3:13,4:12,5:11,6:10,7:9,8:8,9:7,10:6,11:5,12:4,
             13:3,14:2,15:1,16:0,17:26,18:25,19:24,20:23,21:22,22:21,23:20,
             24:19,25:18,26:17,27:27,28:28,29:29,30:30,31:35,32:34,33:33,34:32,35:31,
             36:45,37:44,38:43,39:42,40:47,41:46,42:39,43:38,44:37,45:36,46:41,47:40,
             48:54,49:53,50:52,51:51,52:50,53:49,54:48,55:59,56:58,57:57,58:56,59:55,
             60:64,61:63,62:62,63:61,64:60,65:67,66:66,67:65}
b = FileFilt()
targetdir = projectdir+objeectdir
b.FindFile(dirr=targetdir)
print(b.counter)
for k in b.fileList:
    print k
    if k=='':
        continue
    print k.split("/")[-1]
#    print k.split("/")
    filename_split = k.split("/")[-1].split(".")
    file_name =[]
    if len(filename_split) >1:
 #       print str(filename_split[-2])
        file_name = str(filename_split[-2])
    objdir = targetdir+file_name
    result = mat_load(objdir+'/result/'+file_name)
    BB = result['BB']
    print BB
    BB = result['BB'][0]
    print BB
    result = result['result']

    V= result['vertices'][0,0]
    F = result['faces'][0, 0]
    F = F-1
    output_path = objdir+'/result/'+file_name+'.obj'
#    tmp = F[:,1].copy()
#    F[:, 1] = F[:, 2].copy()
#    F[:, 2] = tmp
    V[:,1] = 192-V[:,1]
    TEST_V = np.array([ [1.2,2.2,3.0],[2.0,3.0,4.0],[2.0,3.0,4.0]])
    V_igl = igl.eigen.MatrixXd(V.astype('float64'))
    F_igl = igl.eigen.MatrixXi(F.astype('intc'))
    VertexNormal = igl.eigen.MatrixXd(V_igl)
    N_vertices = igl.eigen.MatrixXd()
    igl.per_vertex_normals(V_igl,F_igl, igl.PER_VERTEX_NORMALS_WEIGHTING_TYPE_AREA, N_vertices)
    '''
    # Plot the mesh
    viewer = igl.viewer.Viewer()
    viewer.core.show_lines = False
    viewer.data.set_mesh(V_igl,F_igl)
    viewer.data.set_normals(N_vertices)
    viewer.callback_init = viewer_init
    print("Press '1' for per-face normals.")
    print("Press '2' for per-vertex normals.")
    print("Press '3' for per-corner normals.")
    viewer.launch()
    '''
    #convertObj2Mat(objdir + '/Target_cut' + '.obj',objdir + '/Target_cut.mat')
#    convertObj2Mat('D:/mprojects/nricp-master/nricp-master/data' + '/faceSource_align' + '.obj','D:/mprojects/nricp-master/nricp-master/data' + '/faceSource_align.mat',0)
#    convertObj2Mat('D:/mprojects/nricp-master/nricp-master/data' + '/target_scaled' + '.obj',
#                   'D:/mprojects/nricp-master/nricp-master/data' + '/target_scaled.mat',1)
    #convertMat2obj('D:/mprojects/nricp-master/nricp-master/data' + '/faceSource.mat',
     #              'D:/mprojects/nricp-master/nricp-master/data' + '/faceSource' + '.obj')
    convertMat2obj('D:/mprojects/nricp-master/nricp-master/data' + '/Tramsformed.mat',
                   'D:/mprojects/nricp-master/nricp-master/data' + '/Tramsformed' + '.obj','Tramsformed')

    break;
    N_vertices = np.array(N_vertices)
    V_np = np.array(V_igl)

    Target = {'vertices':V,'faces':F+1,'normals':N_vertices}
    Target = {'Target':Target}
    mat_save(Target, objdir + '/Target' + '.mat')
    igl.writeOBJ(objdir + '/Target' + '.obj', igl.eigen.MatrixXd(V_np.astype('float64')),
                    igl.eigen.MatrixXi(F.astype('intc')),igl.eigen.MatrixXd(N_vertices.astype('float64')),
                 igl.eigen.MatrixXi(F.astype('intc')),igl.eigen.MatrixXd(),igl.eigen.MatrixXi())


    break


    mesh_center ,main_dir,eigenvalues =k_main_dir_sklearn(V_np,3)
    main_dir = np.array(main_dir)
    fig = plt.figure(figsize=plt.figaspect(1))
    ax = plt.subplot(111, projection='3d')
    v_sample = random.sample(V_np,V_np.shape[0]/100)
    print 'v_sample',len(v_sample)
    v_sample = np.array(v_sample)
    mesh_center = np.array(mesh_center)
    print 'v_sample', v_sample.shape
    print 'mesh_center',mesh_center
    ax.scatter( (v_sample-mesh_center)[:,0]/192, (v_sample-mesh_center)[:,1]/192, (v_sample-mesh_center)[:,2]/192, c='r')
#    ax.scatter(main_dir[:, 0], main_dir[:, 1], main_dir[:, 2], c='b')
    print 'main_dir', main_dir.shape
    print main_dir
    print 'eigenvalues',eigenvalues
    main_dir[0,:] = main_dir[0,:]*eigenvalues[0]/eigenvalues[0]
    main_dir[1, :] = main_dir[1, :] * eigenvalues[0]/eigenvalues[0]
    main_dir[2, :] = main_dir[2, :] * eigenvalues[2]/eigenvalues[0]
    surf = ax.scatter(main_dir[:,0],main_dir[:,1],main_dir[:,2],c='cyan',alpha =1.0 ,edgecolor= 'r')
    vec_center = np.array([0, 0, 0])
    stack_dir = np.vstack( (vec_center,main_dir[0,:],vec_center,main_dir[1,:],vec_center,main_dir[2,:]))
    print 'stack_dir', stack_dir.shape
    print stack_dir
    ax.plot3D( stack_dir[:,0],stack_dir[:,1],stack_dir[:,2],color='blue')
    ax.set_zlabel('Z')
    ax.set_ylabel('Y')
    ax.set_xlabel('X')
    ax.set_xlim(ax.get_xlim()[::-1])
#    plt.show()
    up_dir = -main_dir[0]
    vertical_normal = -main_dir[1]
    front_dir = main_dir[2]


    ori_image = readImage(objdir+'/'+file_name+'.jpg')
    height,width,dim = ori_image.shape
    landmark_2d = read_landmark(objdir+'/2d/'+file_name+'.txt')
    landmark_2d[:,1] = height -landmark_2d[:,1]
    write_image_and_featurepoint(ori_image,landmark_2d,objdir+'/'+'ori'+'.jpg')
    expand_landmark = np.zeros((landmark_2d.shape[0],3))
    expand_landmark[:,0:2] = landmark_2d[:,0:2]
    expand_landmark[:, 2] = 192
    BB[1] = height -BB[1]-BB[2]
    print 'BB ',BB
    target_V =scaleToOriCoodi_bottomleft(V,BB,192)

    up_dir =up_dir/ np.linalg.norm(up_dir)
    vertical_normal =vertical_normal/ np.linalg.norm(vertical_normal)
    front_dir = front_dir / np.linalg.norm(front_dir)
    mesh_center = np.mean(target_V, axis=0)
    symplane_vertex,symplane_f =sym_plane(front_dir,up_dir,vertical_normal,mesh_center,100)
#    igl.writeOBJ(objdir + '/01_symplane' + '.obj', igl.eigen.MatrixXd(symplane_vertex.astype('float64')),
#                igl.eigen.MatrixXi(symplane_f.astype('intc')))
    #把中间点修正到对称面
    landmark_v = igl.eigen.MatrixXd()
    landmark_f = igl.eigen.MatrixXi()
    igl.readOBJ(objdir + '/01_landmark_cast' + '.obj', landmark_v,
                landmark_f)
    landmark_v = np.array(landmark_v)
    landmark_f = np.array(landmark_f)
    for i  in [27,28,29,30,33,51,62,66,57,8]:
        landmark_v[i,:] =corr_point(front_dir,up_dir,vertical_normal,mesh_center,landmark_v[i,:])
    for i in [9,10,11,12,13,14,15,16,22,23,24,25,26,34,35,42,43,44,45,46,47,52,53,54,55,56,63,64,65]:
        sym_idx = point_sym[i]
        landmark_v[sym_idx, :] =sym_point(front_dir,up_dir,vertical_normal,mesh_center, landmark_v[i,:])

    igl.writeOBJ(objdir + '/01_corr_landmark_cast_sym' + '.obj', igl.eigen.MatrixXd(landmark_v.astype('float64')),
                igl.eigen.MatrixXi(landmark_f.astype('intc')))
    break

    landmark_cast = []
    from time import time
    '''
    timer_start = time()
    for i in range(0,68):
        issucess, result = cast2d_to3d(landmark_2d[i,:], target_V, F)
        if issucess:
            print i,issucess,result
            landmark_cast.append(result)
        else:
            landmark_cast.append([landmark_2d[i,0],landmark_2d[i,1],100])
    timer_end = time()
    print "use time %f sec\n" % ( timer_end - timer_start )
    '''
    landmark_cast = np.array(landmark_cast)

    #插入一个图像背景
    b_v =[[0,0,0],[width,0,0],[0,height,0],[width,height,0]]
    target_V = np.vstack((target_V,b_v))
    num_vertices = V.shape[0]
    target_F = [[num_vertices+0,num_vertices+1,num_vertices+2],[num_vertices+1,num_vertices+3,num_vertices+2]]
    target_F =  np.vstack((F,target_F))
#    igl.writeOBJ(objdir + '/01' + '.obj', igl.eigen.MatrixXd(target_V.astype('float64')), igl.eigen.MatrixXi(target_F.astype('intc')))
    igl.writeOBJ(objdir + '/01_landmark_cast' + '.obj', igl.eigen.MatrixXd(landmark_cast.astype('float64')),
                igl.eigen.MatrixXi(landmark_face.astype('intc')))
    break
#之前通过聚类块方式来求主方向的方法，不如pca
    select_normal =[]
    select_idx =[]
    select_vertex=[]
    select_up_normal =[]
    select_up_idx =[]
    select_up_vertex=[]
    for  i  in range(0,N_vertices.shape[0]):
        cur_normal = N_vertices[i,:]
        dot_result  = cur_normal.dot(np.array([0,0,-1]))
        dot_resultup = cur_normal.dot(np.array([0, 1, 0]))
        if(dot_result>0):
            select_normal.append(cur_normal)
            select_idx.append(i)
            select_vertex.append(V_np[i])
        if(dot_resultup>0.86):
            select_up_normal.append(cur_normal)
            select_up_idx.append(i)
            select_up_vertex.append(V_np[i])
    # face normal
    centroid = kmeans(select_normal, 3)[0]
    print centroid
    label = vq(select_normal, centroid)[0]
    print label.shape
    lable0=[]
    lable1=[]
    label2=[]
    for  i  in range(0,label.shape[0]):
        if label[i] == 0:
            lable0.append(select_normal[i])
        elif label[i] == 1:
            lable1.append(select_normal[i])
        else:
            label2.append(select_normal[i])
    #   print slice
#    igl.writeOBJ(objdir+'/result/'+'lable0'+'.obj', igl.eigen.MatrixXd(select_normal), igl.eigen.MatrixXi())
#    igl.writeOBJ(objdir + '/result/' + 'lable1' + '.obj', igl.eigen.MatrixXd(select_normal), igl.eigen.MatrixXi())
#    igl.writeOBJ(objdir + '/result/' + 'lable2' + '.obj', igl.eigen.MatrixXd(select_normal), igl.eigen.MatrixXi())
    max_label =[]
    if len(lable0) > len(lable1):
        if len(lable0) > len(label2):
            max_label = lable0
        else:
            max_label = label2
    elif  len(lable1) >  len(label2):
        max_label = lable1
    else:
        max_label = label2
    print np.array(max_label).shape
    mean_normal = np.mean(np.array(max_label), axis=0)
    print np.array( [mean_normal,[0,0,0]]).shape
    mean_normal = -mean_normal
    front_normal = mean_normal
    # up normal
    select_normal = select_up_normal
    centroid = kmeans(select_normal, 3)[0]
    print centroid
    label = vq(select_normal, centroid)[0]
    print label.shape
    lable0=[]
    lable1=[]
    label2=[]
    for  i  in range(0,label.shape[0]):
        if label[i] == 0:
            lable0.append(select_normal[i])
        elif label[i] == 1:
            lable1.append(select_normal[i])
        else:
            label2.append(select_normal[i])

    max_label =[]
    if len(lable0) > len(lable1):
        if len(lable0) > len(label2):
            max_label = lable0
        else:
            max_label = label2
    elif  len(lable1) >  len(label2):
        max_label = lable1
    else:
        max_label = label2
    print np.array(max_label).shape
    mean_normal = np.mean(np.array(max_label), axis=0)
    print np.array( [mean_normal,[0,0,0]]).shape
    up_normal = mean_normal
    igl.writeOBJ(objdir+'/result/'+'lable0'+'.obj', igl.eigen.MatrixXd(lable0), igl.eigen.MatrixXi())
    igl.writeOBJ(objdir + '/result/' + 'lable1' + '.obj', igl.eigen.MatrixXd(lable1), igl.eigen.MatrixXi())
    igl.writeOBJ(objdir + '/result/' + 'lable2' + '.obj', igl.eigen.MatrixXd(label2), igl.eigen.MatrixXi())
    print front_normal,up_normal
    '''
    cast_vertex_array = []
    for  i  in range(0,V_np.shape[0]):
        cur_vertex = V_np[i,:]
        dot_result  = cur_vertex.dot(mean_normal)
        cast_vertex = mean_normal*dot_result
        cast_vertex_array.append(cast_vertex)

    print len(cast_vertex_array)
    '''
    #igl.writeOBJ(objdir + '/result/' + 'cast_vertex' + '.obj', igl.eigen.MatrixXd(cast_vertex_array), igl.eigen.MatrixXi())
    #    igl.writeOBJ(output_path,V_igl,F_igl,N_vertices,F_igl,igl.eigen.MatrixXd(),igl.eigen.MatrixXi())
#    write_simple_obj(mesh_v=V, mesh_f=F-1, filepath=output_path, verbose=False)
#    vertexs= vertexs['vertices']
#    np_vertexs = vertexs[0]
#    dir(np_vertexs)
    break

