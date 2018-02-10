# -- coding: utf-8 --
import os
import numpy as np
import cPickle as pickle
import scipy.io as scio
import sys
sys.path.insert(0, "D:/mprojects/libiglfull/libigl/python/build/x64/Release")
import pyigl as igl
from fitting.util import write_full_obj,mat_load,readVertexColor
from triangle_raster import  FP_COLOR_TO_TEXTURE,MetroMesh
from fitting.util import load_binary_pickle, save_binary_pickle
import cv2

'''
para_objmesh： 需要有参数化的坐标
paraTexMat： 用于得到 顶点坐标
'''

def conver_mean_tex(para_objmesh, paraTexMat,outputdir):
    v = igl.eigen.MatrixXd()
    f = igl.eigen.MatrixXi()
    n = igl.eigen.MatrixXd()
    n_f = igl.eigen.MatrixXi()
    t = igl.eigen.MatrixXd()
    t_f = igl.eigen.MatrixXi()
    igl.readOBJ(para_objmesh, v, t, n, f, t_f, n_f)
    v = np.array(v)
    t = np.array(t)
    n = np.array(n)
    f = np.array(f)
    t_f = np.array(t_f)
    n_f = np.array(n_f)
    contact = mat_load(paraTexMat)
    target = contact['paraTex']
    texMU = target['texMU'][0, 0]
    cur_tex = texMU

    for i in range(0, cur_tex.size):
        if cur_tex[i] < 0.0:
            cur_tex[i] = 0.0
        if cur_tex[i] > 255.0:
            cur_tex[i] = 255.0
        cur_tex[i] /= 255.0
    cur_tex = cur_tex.reshape(cur_tex.size / 3, 3)
    output_path = outputdir + 'mean_tex' + '.obj'
    write_full_obj(v, f, n, n_f, t, t_f, cur_tex, output_path)

def convert_para_tex(para_objmesh, paraTexMat,outputdir,ev_std=5):

    v = igl.eigen.MatrixXd()
    f = igl.eigen.MatrixXi()
    n = igl.eigen.MatrixXd()
    n_f = igl.eigen.MatrixXi()
    t = igl.eigen.MatrixXd()
    t_f = igl.eigen.MatrixXi()
    igl.readOBJ(para_objmesh, v,t,n,f,t_f,n_f)
    v = np.array(v)
    t = np.array(t)
    n = np.array(n)
    f = np.array(f)
    t_f = np.array(t_f)
    n_f = np.array(n_f)
    contact = mat_load(paraTexMat)
    target = contact['paraTex']
    texPC = target['texPC'][0, 0]
    texMU = target['texMU'][0, 0]
    texEV = target['texEV'][0, 0]
    for i_pc in range(0,len(texEV)):

        def dir(plus=True):
            coff = np.zeros((len(texEV),1))
            if plus:
                coff[i_pc] =texEV[i_pc]*ev_std  #使用5倍的方差
            else:
                coff[i_pc] = -texEV[i_pc] * ev_std  # 使用5倍的方差
            coff.reshape(coff.size,1)
            cur_tex = texMU+np.dot(texPC,coff)
            for i in range(0,cur_tex.size):
                if cur_tex[i]<0.0:
                    cur_tex[i] =0.0
                if cur_tex[i] > 255.0:
                    cur_tex[i] = 255.0
                cur_tex[i]/=255.0
            cur_tex = cur_tex.reshape(cur_tex.size/3,3)
            if plus:
                output_path = outputdir + 'pc_' + str(i_pc).zfill(3) + 'std_'+str(ev_std) + '+'+'.obj'
            else:
                output_path = outputdir + 'pc_' + str(i_pc).zfill(3) + 'std_' + str(ev_std) + '-' + '.obj'
            write_full_obj(v,f,n,n_f,t,t_f,cur_tex,output_path)

        dir(True)
        dir(False)

def generate_frame_para_tex(obj_path,outfile):
    V_igl = igl.eigen.MatrixXd()
    t_v = igl.eigen.MatrixXd()
    n_v = igl.eigen.MatrixXd()
    v_f = igl.eigen.MatrixXi()
    n_f = igl.eigen.MatrixXi()
    t_f = igl.eigen.MatrixXi()

    igl.readOBJ(obj_path, V_igl, t_v, n_v,
                v_f, t_f, n_f)
    np_v = np.array(V_igl)
    np_t_v = np.array(t_v)
    np_n_v = np.array(n_v)
    np_v_f = np.array(v_f)
    np_n_f = np.array(n_f)
    np_t_f = np.array(t_f)
    np_v = np_v[:, 0:3]
    vertex_color = readVertexColor(obj_path)

    vertex_color[:, :] = (vertex_color[:, :] * 255)

    mesh = MetroMesh()
#    mesh.set_mesh(v=np_v, vertex_color=vertex_color, normal=np_n_v,
#                  vt=np_t_v, face=np_v_f, n_face=np_v_f, t_face=np_t_f)
    mesh.set_mesh(np_v, vertex_color, np_n_v,
                  np_t_v, np_v_f, np_v_f, np_t_f)
    FP_COLOR_TO_TEXTURE(outfile, mesh, 1024, 1024)

def generate_para_frame_obj(frame_para_objmesh,outputdir):

    v = igl.eigen.MatrixXd()
    f = igl.eigen.MatrixXi()
    n = igl.eigen.MatrixXd()
    n_f = igl.eigen.MatrixXi()
    t = igl.eigen.MatrixXd()
    t_f = igl.eigen.MatrixXi()
    igl.readOBJ(frame_para_objmesh, v, t, n, f, t_f, n_f)
    v = np.array(v)
    t = np.array(t)
    n = np.array(n)
    f = np.array(f)
    t_f = np.array(t_f)
    n_f = np.array(n_f)

    ev_std =5
    for i_pc in range(0,199):
        output_path = outputdir +'frame_pc_' + str(i_pc).zfill(3) + 'std_' + str(ev_std) + '+' + '.obj'
        write_full_obj(v, f, n, n_f, t, t_f, np.array([]), output_path,generate_mtl=True,verbose=False,
                       img_name = 'pc_' + str(i_pc).zfill(3) + 'std_' + str(ev_std) + '+'+ '.png')
        output_path = outputdir + 'frame_pc_' + str(i_pc).zfill(3) + 'std_' + str(ev_std) + '-' + '.obj'
        write_full_obj(v, f, n, n_f, t, t_f, np.array([]), output_path, generate_mtl=True, verbose=False,
                       img_name='pc_' + str(i_pc).zfill(3) + 'std_' + str(ev_std) + '-' + '.png')

def build_para_tex_pc(img_path,paraTexMat,outpudir):
    meanimagepath = img_path+'mean_tex_color.png'
    mean_img = cv2.imread(meanimagepath,cv2.IMREAD_COLOR )
    ev_std = 5
    tx_pc = np.zeros((1,mean_img.shape[0],mean_img.shape[1],3))
    for i_pc in range(0, 1):
        'pc_' + str(i_pc).zfill(3) + 'std_' + str(ev_std) + '+' + '.png'
        pc_img = cv2.imread(meanimagepath, cv2.IMREAD_COLOR)
        c_pc = (pc_img[:,:,:]-mean_img[:,:,:])/ev_std
        tx_pc[i_pc,:,:,:]= c_pc[:,:,:]
    contact = mat_load(paraTexMat)
    target = contact['paraTex']
    texEV = target['texEV'][0, 0]
    texEV = np.array(texEV)
    data = {'texPC':tx_pc,'texMU':mean_img,'texEV':texEV}
    save_binary_pickle(data,outpudir)



def read_mask(mask_image_path,whiteIsMask = True):
    img = cv2.imread(mask_image_path,cv2.IMREAD_GRAYSCALE )
    img_mask = np.zeros((img.shape[0],img.shape[1]),np.bool)
    img_mask[:,:] = False
    for i in range(0,img.shape[0]):
        for j in range(0,img.shape[1]):
            if whiteIsMask:
                if img[i,j] == 255:
                    img_mask[i,j] = True

            else:
                if img[i,j] == 0:
                    img_mask[i,j] = True

def convert2Texture(inputdir,outputdir):
    ev_std =5
    for i_pc in range(145,199):
        input_path = inputdir + 'pc_' + str(i_pc).zfill(3) + 'std_' + str(ev_std) + '+' + '.obj'
        output_path = outputdir + 'pc_' + str(i_pc).zfill(3) + 'std_' + str(ev_std) + '+'+ '.png'
        generate_frame_para_tex(input_path,output_path)
        #input_path = inputdir + 'pc_' + str(i_pc).zfill(3) + 'std_' + str(ev_std) + '-' + '.obj'
        #output_path = outputdir + 'pc_' + str(i_pc).zfill(3) + 'std_' + str(ev_std) + '-'+ '.png'
        #generate_frame_para_tex(input_path, output_path)

def generate_tex():
    source_mesh = 'D:/mprojects/flame-fitting/texpc/source2.obj'
    conver_mean_tex(source_mesh,'D:/mprojects/flame-fitting/paraTex.mat','D:/mprojects/flame-fitting/texpc/source_para/')
    #convert_para_tex(source_mesh,'D:/mprojects/flame-fitting/paraTex.mat','D:/mprojects/flame-fitting/texpc/source_para/',ev_std=5)
#    mean_tex_img = cv2.imread('D:/mprojects/flame-fitting/texpc/mean_tex_color_witheye.png', cv2.IMREAD_UNCHANGED)
#    eye_mask = read_mask('D:/mprojects/flame-fitting/texpc/eye_mask.png',whiteIsMask = False)

def add_texture_and_write(v_np,f_np,output_path):
    from  fitting.util import cac_normal,read_igl_obj
    normal_np = cac_normal(v_np,f_np)
    normal_f = f_np
    source_mesh =  'D:/mprojects/flame-fitting/texpc/target_para/frame_pc_000std_5+.obj'
    v, f, t, t_f, n, n_f = read_igl_obj(source_mesh)
    write_full_obj(v_np, f, normal_np, n_f, t, t_f, np.array([]), output_path, generate_mtl=True, verbose=False,
                   img_name='pc_' + str(0).zfill(3) + 'std_' + str(5) + '+' + '.png')


if __name__ == '__main__':
    #generate_tex()
#    convert2Texture('D:/mprojects/flame-fitting/texpc/source_para/','D:/mprojects/flame-fitting/texpc/source_para/texture/')
    #generate_para_frame_obj('D:/mprojects/flame-fitting/texpc/frame_template_witheye.obj',
    #                        'D:/mprojects/flame-fitting/texpc/target_para/')
    build_para_tex_pc('D:/mprojects/flame-fitting/texpc/source_para/texture/', 'D:/mprojects/flame-fitting/paraTex.mat',
                     'D:/mprojects/flame-fitting/texpc/target_para/frame_tex.pkl')
    pass