# -- coding: utf-8 --

import sys
sys.path.insert(0, "D:/mprojects/libiglfull/libigl/python/build/x64/Release")
import pyigl as igl
from os.path import join

from fitting.util import safe_mkdir, mat_save,mat_load,IglMatrixTonpArray,FileFilt,write_simple_obj,k_main_dir_sklearn
from fitting.util import readImage,write_image_and_featurepoint,read_landmark,\
    cast2d_to3d,scaleToOriCoodi_bottomleft,sym_plane,corr_point,sym_point,convertObj2Mat,convertMat2obj
'''
convertObj2Mat('D:/mprojects/face_m_files/final_result' + '/scan_face_nrd_cut' + '.obj',
                  'D:/mprojects/face_m_files/final_result' + '/target_scaled.mat',1)
convertObj2Mat('D:/mprojects/face_m_files/final_result' + '/template_face_ra' + '.obj',
                  'D:/mprojects/face_m_files/final_result' + '/source_face.mat',0)

convertObj2Mat('D:/mprojects/face_m_files/final_result/rigid_aligned' + '/scan_target' + '.obj',
                  'D:/mprojects/face_m_files/final_result/rigid_aligned' + '/target_scaled.mat',1)
convertObj2Mat('D:/mprojects/face_m_files/final_result/rigid_aligned' + '/template_source' + '.obj',
                  'D:/mprojects/face_m_files/final_result/rigid_aligned' + '/source_face.mat',0)

convertMat2obj( 'D:/mprojects/face_m_files/final_result/rigid_aligned' + '/source_face.mat','D:/mprojects/face_m_files/final_result/rigid_aligned' + '/scan_target' + '.obj',
                    'Source')
'''
from fitting.util import determinant,cast_coeff
from fitting.util import convert_para_tex,process_maya_select_vtx,save_int
#convert_para_tex('./uv_mapping/bwm/rnd_seg_head_regen.obj','./paraTex.mat')
#result = process_maya_select_vtx('D:/mprojects/flame-fitting/face_front_mask_maya.txt')
#result =result.reshape(result.size,1)
#save_int('D:/mprojects/flame-fitting/face_front_mask.txt',result)
'''
import cv2
image = cv2.imread('D:/mprojects/flame-fitting/uv_mapping/bwm/example/crop.png',cv2.IMREAD_COLOR)
type(image)
print image
cv2.imwrite('D:/mprojects/flame-fitting/uv_mapping/bwm/example/test_cvwrite.jpg', image)
'''
prj_dir = 'D:/mprojects/flame-fitting/uv_mapping/bwm/example/'
prj_dir = 'D:/mprojects/flame-fitting/uv_mapping/bwm/test_simplication/'
#prj_dir = 'D:/mprojects/flame-fitting/uv_mapping/bwm/test_simplication/test_transfer/'
#prj_dir = 'D:/mprojects/flame-fitting/uv_mapping/bwm/test_simplication/test_transfer/aligned/'

from triangle_raster import  FP_COLOR_TO_TEXTURE

from triangle_raster import   MetroMesh
from fitting.util import readVertexColor,write_full_obj,write_landmark_to_obj
import sys
sys.path.insert(0, "D:/mprojects/libiglfull/libigl/python/build/x64/Release")
import pyigl as igl
import numpy as np
V_igl = igl.eigen.MatrixXd()
t_v = igl.eigen.MatrixXd()
n_v = igl.eigen.MatrixXd()
v_f = igl.eigen.MatrixXi()
n_f = igl.eigen.MatrixXi()
t_f = igl.eigen.MatrixXi()

objname = 'rnd_seg_head_regen_simp' #'rnd_seg_head_regen' #'rnd_seg_head_regen_simp_nouv' #'head_has_uv'
igl.readOBJ(prj_dir + '/'+ objname+ '.obj', V_igl, t_v, n_v,
            v_f, t_f, n_f)
np_v = np.array(V_igl)
np_t_v = np.array(t_v)
np_n_v = np.array(n_v)
np_v_f  = np.array(v_f)
np_n_f  = np.array(n_f)
np_t_f  = np.array(t_f)
np_v = np_v[:,0:3]
np_v = np_v/100000.0
vertex_color = readVertexColor(prj_dir + '/'+ objname + '.obj')

#hhh = np.load('fp_index.npz')
#fp_index = hhh['fp_index']
#landmark = np_v[fp_index,:]
# write_landmark_to_obj(prj_dir+objname+
#                '_normalized_landmark'+'.obj',landmark,size=10.0)
# write_full_obj(np_v,np_v_f,np_n_v,np_n_f,np_t_v,np_t_f,vertex_color,prj_dir+objname+
#                '_normalized'+'.obj')




vertex_color[:,:] = (vertex_color[:,:]*255)

mesh = MetroMesh()
mesh.set_mesh(v=np_v,vertex_color=vertex_color,normal=np_n_v,
               vt=np_t_v,face=np_v_f,n_face =np_v_f,t_face=np_t_f)

FP_COLOR_TO_TEXTURE(prj_dir+'rnd_seg_head_regen_simp_pp.png',mesh,1024,1024)

#from z_buffer_raster import Mesh_render_to_image
#Mesh_render_to_image(prj_dir+'head_has_uv_python_render.png',mesh,1024,1024)


