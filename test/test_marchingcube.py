# -- coding: utf-8 --
import numpy as np
def test_ellipsoid():
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    from skimage import measure
    from skimage.draw import ellipsoid

    from fitting.util import write_full_obj
    # Generate a level set about zero of two identical ellipsoids in 3D
    ellip_base = ellipsoid(6, 10, 16, levelset=False)
    ellip_double = np.concatenate((ellip_base[:-1, ...],
                                   ellip_base[2:, ...]), axis=0)

    # Use marching cubes to obtain the surface mesh of these ellipsoids
    verts, faces, normals, values = measure.marching_cubes_lewiner(ellip_base, 0)
    write_full_obj(mesh_v=verts, mesh_f=faces, mesh_n=np.array([]),mesh_n_f=np.array([]),mesh_tex=np.array([]),mesh_tex_f=np.array([]),
                   vertexcolor=np.array([]),filepath='L:\yuanqing/marching_cubes_lewiner.obj', generate_mtl=False,verbose=False,img_name = 'default.png')

    # Display resulting triangular mesh using Matplotlib. This can also be done
    # with mayavi (see skimage.measure.marching_cubes_lewiner docstring).
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces])
    mesh.set_edgecolor('k')
    ax.add_collection3d(mesh)

    ax.set_xlabel("x-axis: a = 6 per ellipsoid")
    ax.set_ylabel("y-axis: b = 10")
    ax.set_zlabel("z-axis: c = 16")

    ax.set_xlim(0, 24)  # a = 6 (times two for 2nd ellipsoid)
    ax.set_ylim(0, 20)  # b = 10
    ax.set_zlim(0, 32)  # c = 16

    plt.tight_layout()
    plt.show()

def test2():
    from fitting.util import write_full_obj
    from skimage import measure
    grid = np.zeros((100,100,100),np.bool)
    grid = np.zeros((100,100,100), np.float)
    grid[:, :, :] = 1.0
    grid[50:70,50:70,50:70] = -1.0

    verts, faces, normals, values = measure.marching_cubes_lewiner(grid, 0)
    write_full_obj(mesh_v=verts, mesh_f=faces, mesh_n=np.array([]),mesh_n_f=np.array([]),mesh_tex=np.array([]),mesh_tex_f=np.array([]),
                   vertexcolor=np.array([]),filepath='E:\workspace\dataset\hairstyles\hair/test_hair_wrapper/hair.obj', generate_mtl=False,verbose=False,img_name = 'default.png')

if __name__ == '__main__':
    test2()
    pass