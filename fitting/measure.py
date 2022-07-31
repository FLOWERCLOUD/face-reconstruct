# -- coding: utf-8 --
'''
Util funcitons for measurement
Tianye Li <tianye.li@tuebingen.mpg.de>
'''

import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np


# -----------------------------------------------------------------------------

def mesh2mesh( mesh_v_1, mesh_v_2 ):
    dist = np.linalg.norm(mesh_v_2 - mesh_v_1, axis=1)
    return dist
'''
mesh_v1 - mesh_v2
'''
def signdistance(mesh_v1,mesh_v2,mesh_v2_noraml):
    dist = np.linalg.norm(mesh_v1 - mesh_v2, axis=1)
    dis_vector = mesh_v1 - mesh_v2
    if dist.shape[0] != mesh_v2_noraml.shape[0]:
        print 'normal shape not correct'
    for i in range(0,mesh_v2_noraml.shape[0]):
        if np.dot(dis_vector[i,:],mesh_v2_noraml[i,:]) <0:
            dist[i] =-dist[i]
    return dist

# -----------------------------------------------------------------------------

def distance2color( dist, vmin=0, vmax=0.001, cmap_name='jet' ):
    # vmin, vmax in meters
    norm = mpl.colors.Normalize( vmin=vmin, vmax=vmax )
    cmap = cm.get_cmap( name=cmap_name )
    colormapper = cm.ScalarMappable( norm=norm, cmap=cmap )
    rgba = colormapper.to_rgba( dist )
    color_3d = rgba[:,0:3]
    return color_3d

def mesh_error_compare2(input_mesh_path1,input_mesh_path2,output_mesh_dir,index):
    from  face_generate import get_z_buffer
    from util import read_igl_obj,get_mean_value,get_vertex_normal,bacface_cull
    import cv2
    subdived_mesh, subdived_mesh_f, t_frame_aligned, t_f_frame_aligned, n_frame_aligned, n_f_frame_aligned =read_igl_obj(input_mesh_path1)
    subdived_mesh2, subdived_mesh_f2, t_frame_aligned2, t_f_frame_aligned2, n_frame_aligned2, n_f_frame_aligned2 = read_igl_obj(input_mesh_path2)
    img = np.zeros((500,400),np.int)
    min1 = np.array([-1.1,-1.1/400*500,-1.6])
    max1 = np.array([1.1,1.1/400*500,1.0])
    translate =np.array([-min1[0],-min1[1],-min1[2]])
    scale = 400/2.2
    V1 = (subdived_mesh+translate)*scale
    V2 = (subdived_mesh2 + translate) * scale
    F1 = subdived_mesh_f
    F2 =subdived_mesh_f2
    N1 = get_vertex_normal(V1,F1)
    N2 = get_vertex_normal(V2, F2)
    V1,F1 = bacface_cull(V1,F1,N1)
    V2, F2 = bacface_cull(V2, F2, N2)
    render_img, z_buffer_img = get_z_buffer(V1,F1,img,output_render_img=output_mesh_dir+'/1.png',vcolor = np.array([]))
    render_img2, z_buffer_img2 = get_z_buffer(V2, F2, img, output_render_img=output_mesh_dir+'/2.png', vcolor=np.array([]))
    z1 = z_buffer_img / 2.0
    z2 = z_buffer_img2 / 2.0
    print z1.min(),z1.max()
    print z2.min(), z2.max()
    dis = abs(z2-z1)/z1
    isvalid = np.zeros((500, 400), np.bool) #if the pixel valid
    isvalid[:,:] =True
    for j in range(0,dis.shape[0]):
        for i in range(0,dis.shape[1]):
            if z_buffer_img[j,i] <0:
                isvalid[j,i] =False
            if z_buffer_img2[j,i] <0:
                isvalid[j,i] =False
    #以第一个为groud truth
    # for j in range(0,dis.shape[0]):
    #     for i in range(0,dis.shape[1]):
    #         if z_buffer_img[j,i] <0:
    #             isvalid[j,i] =False

    valid_dis =[]
    for j in range(0,dis.shape[0]):
        for i in range(0,dis.shape[1]):
            if isvalid[j,i] == False:
                dis[j,i] = 0.0
            else:
                valid_dis.append(dis[j,i])
    v_mean,v_var,std_var = get_mean_value(valid_dis)
    print v_mean,'+-',std_var
    vmax = z1.max()*0.5
    vmax = dis.max()*10
    vmax = 0.5
    ori_height = dis.shape[0]
    ori_width = dis.shape[1]
    dis = dis.reshape(ori_height*ori_width)
    color_3d = distance2color(dist=dis, vmin=0, vmax=vmax, cmap_name='jet')
    color_3d = color_3d.reshape((ori_height,ori_width,3))
    cv2.imwrite(output_mesh_dir + '/'+str(index).zfill(3)+'_1z.png',z1)
    cv2.imwrite(output_mesh_dir+ '/'+str(index).zfill(3)+'_2z.png',z2)
    color_3d = color_3d[:, :, ::-1]
    cv2.imwrite(output_mesh_dir + '/'+str(index).zfill(3)+'_cz.png', color_3d*255)

if __name__ == '__main__':
    import os,shutil
    # mesh_error_compare2(
    #     input_mesh_path1='D:\huayunhe/facewarehouse_new\FaceWarehouse_neutral_img_obj//triangle/Tester_1_pose_0.obj',
    #     input_mesh_path2='D:\huayunhe/facewarehouse_new\FaceWarehouse_neutral_img_obj/align/align000.obj',
    #     output_mesh_dir='D:\huayunhe/facewarehouse_new\FaceWarehouse_neutral_img_obj/')
    if 0: #
        female_path = 'D:/huayunhe/facewarehouse_new/FaceWarehouse_neutral_img_b/female/'
        male_path = 'D:/huayunhe/facewarehouse_new/FaceWarehouse_neutral_img_b/male/'
        for i in range(1,151):
            obj_path = str(i).zfill(3)
            target_path = 'Tester_'+obj_path+'_pose_0'
            all_path1 = female_path+target_path+'/generate_face/'+'face_with_texture.obj'
            all_path2 = male_path+target_path+'/generate_face/'+'face_with_texture.obj'
            if os.path.exists(all_path1):
                dstfile = female_path+target_path+'/generate_face/'+target_path+'.obj'
                shutil.copyfile(all_path1, dstfile)
                pass
            elif os.path.exists(all_path2):
                dstfile = male_path+target_path+'/generate_face/'+target_path+'.obj'
                shutil.copyfile(all_path2, dstfile)
                pass
        pass
    if 0:
        for i in range(0, 150):
            #all_path= 'D:/huayunhe/facewarehouse_new/FaceWarehouse_neutral_img_b/all/'+'Tester_'+str(i+1).zfill(3)+'_pose_0'+'.obj'
            align_path = 'D:/huayunhe/facewarehouse_new/FaceWarehouse_neutral_img_obj/align/alignk'+str(i).zfill(3)+'.obj'
            print 'index',i+1
            mesh_error_compare2(
                input_mesh_path1='D:\huayunhe/facewarehouse_new\FaceWarehouse_neutral_img_obj//triangle/Tester_'+str(i+1)+'_pose_0.obj',
                input_mesh_path2=align_path,
                output_mesh_dir='D:\huayunhe/facewarehouse_new\FaceWarehouse_neutral_img_obj/output/',index = i+1)

    mesh_error_compare2(
                input_mesh_path1='D:\huayunhe/facewarehouse_new\FaceWarehouse_neutral_img_obj//triangle/Tester_'+str(i+1)+'_pose_0.obj',
                input_mesh_path2=align_path,
                output_mesh_dir='D:\huayunhe/facewarehouse_new\FaceWarehouse_neutral_img_obj/output/',index = i+1)
    