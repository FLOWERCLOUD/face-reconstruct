import sys
import numpy as np
import trimesh
from fitting.util import save_landmark
# print logged messages
trimesh.util.attach_to_log()
from configs.config import igl_python_path
sys.path.insert(0, igl_python_path)
import pyigl as igl
# load a mesh
mesh = trimesh.load('L:/yuanqing/imgs/imgs/vrn_result/niutou/01/01.obj')
ray_origins = np.zeros((1024*683,3),dtype=np.int)
ray_directions =  np.zeros((1024*683,3),dtype=np.int)
for x in range (0,1024):
    for y in range(0,683):
        ray_origins[x*683+y,:] = np.array([x,y,192])
        ray_directions[x*683+y,:] = np.array([0, 0, -1])

index_triangles, index_ray,result_locations = mesh.ray.intersects_id(ray_origins=ray_origins,
                                                         ray_directions=ray_directions,
                                                         multiple_hits=False,return_locations=True)
print index_triangles,index_ray,result_locations
index_triangles = index_triangles.reshape(index_triangles.size,1)
index_ray = index_triangles.reshape(index_ray.size,1)
save_landmark('index_triangles.txt',index_triangles)
save_landmark('index_ray.txt',index_ray)
save_landmark('result_locations.txt',result_locations)

igl.writeOBJ('embree_hit' + '.obj', igl.eigen.MatrixXd(result_locations.astype('float64')),
             igl.eigen.MatrixXi(), igl.eigen.MatrixXd(),
             igl.eigen.MatrixXi(), igl.eigen.MatrixXd(), igl.eigen.MatrixXi())