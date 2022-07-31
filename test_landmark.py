# -- coding: utf-8 --

import sys
sys.path.insert(0, "D:/mprojects/libiglfull/libigl/python/build/x64/Release")
import pyigl as igl
from fitting.landmarks import load_embedding
from smpl_webuser.serialization import load_model
import numpy as np
# landmark embedding
lmk_emb_path = './data/lmk_embedding_intraface_to_flame.pkl'
lmk_face_idx, lmk_b_coords = load_embedding(lmk_emb_path)
model_path = './models/female_model.pkl'  # change to 'female_model.pkl' or 'generic_model.pkl', if needed
model = load_model(
    model_path)

sphere_v = igl.eigen.MatrixXd()
sphere_f = igl.eigen.MatrixXi()
igl.readOBJ('sphere.obj', sphere_v,
            sphere_f)
sphere_v = np.array(sphere_v)
sphere_f = np.array(sphere_f)


model.pose[6:9] = np.array([0.2,0,0])
# select corresponding vertices
mesh_verts = model.r
mesh_faces =  model.f
embedding_vertex = np.vstack([(mesh_verts[mesh_faces[lmk_face_idx], 0] * lmk_b_coords).sum(axis=1),
                  (mesh_verts[mesh_faces[lmk_face_idx], 1] * lmk_b_coords).sum(axis=1),
                  (mesh_verts[mesh_faces[lmk_face_idx], 2] * lmk_b_coords).sum(axis=1)]).T

lmk_num = embedding_vertex.shape[0]
igl.writeOBJ('./landmark_embedding_squence_openmouse/' +'mesh' + '.obj', igl.eigen.MatrixXd(mesh_verts.astype('float64')), igl.eigen.MatrixXi(mesh_faces.astype('intc')))
for i in range(0,lmk_num):
    sphere_v_move = sphere_v+embedding_vertex[i,:]
    igl.writeOBJ('./landmark_embedding_squence_openmouse/' + str(i) + '.obj', igl.eigen.MatrixXd(sphere_v_move.astype('float64')), igl.eigen.MatrixXi(sphere_f.astype('intc')))

frame_landmark_idx = range(17,60)+range(61,64)+range(65,68)


