# -- coding: utf-8 --
"""
Util funcitons - general
Tianye Li <tianye.li@tuebingen.mpg.de>
"""

import os
import numpy as np
import cPickle as pickle
import scipy.io as scio
import sys
from configs.config import igl_python_path

sys.path.insert(0, igl_python_path)
import pyigl as igl
import math


# -----------------------------------------------------------------------------

def load_binary_pickle(filepath):
    # type: (object) -> object
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data


def load_pickle(filepath):
    # type: (object) -> object
    with open(filepath, ) as f:
        data = pickle.load(f)
    return data


# -----------------------------------------------------------------------------

def save_binary_pickle(data, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)


# -----------------------------------------------------------------------------

def load_simple_obj(filename):
    f = open(filename, 'r')

    def get_num(string, type):
        if type == 'int':
            return int(string)
        elif type == 'float':
            return float(string)
        else:
            print 'Wrong type specified'

    vertices = []
    faces = []

    for line in f:
        str = line.split()
        if len(str) == 0:
            continue

        if str[0] == '#':
            continue
        elif str[0] == 'v':
            tmp_v = [get_num(s, 'float') for s in str[1:]]
            vertices.append(tmp_v)

        elif str[0] == 'f':
            tmp_f = [get_num(s, 'int') - 1 for s in str[1:]]
            faces.append(tmp_f)

    f.close()
    return (np.asarray(vertices), np.asarray(faces))


# -----------------------------------------------------------------------------

def write_simple_obj(mesh_v, mesh_f, filepath, verbose=False):
    with open(filepath, 'w') as fp:
        for v in mesh_v:
            fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
        for f in mesh_f:  # Faces are 1-based, not 0-based in obj files
            fp.write('f %d %d %d\n' % (f[0] + 1, f[1] + 1, f[2] + 1))
    if verbose:
        print 'mesh saved to: ', filepath


def write_full_obj(mesh_v, mesh_f, mesh_n, mesh_n_f, mesh_tex, mesh_tex_f, vertexcolor, filepath, generate_mtl=False,
                   verbose=False, img_name='default.png'):
    with open(filepath, 'w') as fp:

        if generate_mtl:
            mtlfile = filepath + '.mtl'
            with open(mtlfile, 'w') as fmtl:
                fmtl.write('newmtl material_0\n')
                fmtl.write('Ka 0.200000 0.200000 0.200000\n')
                fmtl.write('Kd 0.000000 0.000000 0.000000\n')
                fmtl.write('Ks 1.000000 1.000000 1.000000\n')
                fmtl.write('Tr 0.000000\n')
                fmtl.write('illum 2\n')
                fmtl.write('Ns 0.000000\n')
                fmtl.write('map_Kd ' + img_name + '\n')
            mtlname = mtlfile.split('/')[-1]
            fp.write('mtllib ' + './' + mtlname + '\n')
        count_v = 0
        for v in mesh_v:
            if vertexcolor.size > 0:
                fp.write('v %f %f %f %f %f %f\n' % (v[0], v[1], v[2],
                                                    vertexcolor[count_v, 0], vertexcolor[count_v, 1],
                                                    vertexcolor[count_v, 2]))
            else:
                fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
            count_v += 1
        for v in mesh_tex:
            fp.write('vt %f %f\n' % (v[0], v[1]))
        for v in mesh_n:
            fp.write('vn %f %f %f\n' % (v[0], v[1], v[2]))
        count_f = 0
        for f in mesh_f + 1:  # Faces are 1-based, not 0-based in obj files
            fp.write('f %d/' % (f[0]))
            if mesh_tex_f.size > 0:
                fp.write('%d' % (mesh_tex_f[count_f, 0] + 1))
            else:
                pass
            fp.write('/')
            if mesh_n_f.size > 0:
                fp.write('%d' % (mesh_n_f[count_f, 0] + 1))
            else:
                pass
            fp.write(' ')
            fp.write('%d/' % (f[1]))
            if mesh_tex_f.size > 0:
                fp.write('%d' % (mesh_tex_f[count_f, 1] + 1))
            else:
                pass
            fp.write('/')
            if mesh_n_f.size > 0:
                fp.write('%d' % (mesh_n_f[count_f, 1] + 1))
            else:
                pass
            fp.write(' ')
            fp.write('%d/' % (f[2]))
            if mesh_tex_f.size > 0:
                fp.write('%d' % (mesh_tex_f[count_f, 2] + 1))
            else:
                pass
            fp.write('/')
            if mesh_n_f.size > 0:
                fp.write('%d' % (mesh_n_f[count_f, 2] + 1))
            else:
                pass
            fp.write('\n')

            count_f += 1
        if verbose:
            print 'mesh saved to: ', filepath

        # -----------------------------------------------------------------------------


def readVertexColor(path):
    result = []
    with open(path, 'r') as f:
        data = f.readlines()  # txt中所有字符串读入data
        for line in data:
            odom = line.split(' ')  # 将单个数据分隔开存好
            if len(odom) > 0 and odom[0] == 'v':
                if len(odom) > 6:
                    c1 = float(odom[4])  #
                    c2 = float(odom[5])  #
                    c3 = float(odom[6])  #
                    result.append([c1, c2, c3])
    return np.array(result)


def safe_mkdir(file_dir):
    if not os.path.exists(file_dir):
        os.mkdir(file_dir)


# 这个可以逐级创建
def safe_mkdirs(file_dir):
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)


def pickle_save(contact, filepath):  # 使用pickle模块将数据对象保存到文件
    f = open(filepath, 'w')
    pickle.dump(contact, f)
    f.close()


def pickle_load(filepath):  # 使用pickle从文件中重构python对象
    f = open(filepath, 'r')
    contact = pickle.load(f)
    f.close()
    return contact


def mat_save(contact, filepath):  # 存储为mat文件
    scio.savemat(filepath, contact)


def convertObj2Mat(objpath, matpath, istarget):
    v = igl.eigen.MatrixXd()
    f = igl.eigen.MatrixXi()
    n = igl.eigen.MatrixXd()
    n_f = igl.eigen.MatrixXi()
    t = igl.eigen.MatrixXd()
    t_f = igl.eigen.MatrixXi()
    igl.readOBJ(objpath, v, t, n, f, t_f, n_f)
    v = np.array(v)
    f = np.array(f)
    n = np.array(n)
    if istarget:
        Target = {'vertices': v, 'faces': f + 1, 'normals': n}
        Target = {'Target': Target}
        mat_save(Target, matpath)
    else:
        Source = {'vertices': v, 'faces': f + 1, 'normals': n}
        Source = {'Source': Source}
        mat_save(Source, matpath)


def convertMat2obj(matpath, objpath, str):
    contact = mat_load(matpath)
    target = contact[str]
    vertices = target['vertices'][0, 0]
    faces = target['faces'][0, 0]
    normals = target['normals'][0, 0]
    faces = faces - 1
    igl.writeOBJ(objpath, igl.eigen.MatrixXd(vertices.astype('float64')),
                 igl.eigen.MatrixXi(faces.astype('intc')), igl.eigen.MatrixXd(normals.astype('float64')),
                 igl.eigen.MatrixXi(faces.astype('intc')), igl.eigen.MatrixXd(), igl.eigen.MatrixXi())


def mat_load(filepath):  # 读取mat文件
    contact = scio.loadmat(filepath, struct_as_record=True)
    return contact


def npArrayToIglMatrixi(input):
    e2 = igl.eigen.MatrixXi()
    e2.resize(input.shape[0], input.shape[1])
    for i in range(0, input.shape[0]):
        for j in range(0, input.shape[1]):
            e2[i, j] = input[i][j]
    return e2


def IglMatrixTonpArray(input):
    # e2 = np.arange(input.shape[0]*input.shape[1]).reshape(input.shape[0], input.shape[1])
    e2 = np.array(input)
    '''
    for i in range(0, input.shape[0]):
        for j in range(0, input.shape[1]):
            e2[i][j] = input[i, j]
    '''
    return e2


# https://www.jianshu.com/p/4528aaa6dc48
def k_main_dir(Mesh, K):
    mean = np.mean(Mesh)
    res = Mesh - mean


from sklearn.decomposition import PCA


def k_main_dir_sklearn(Mesh, K=3):
    n_samples = Mesh.shape[0]
    pca = PCA()
    X_transformed = pca.fit_transform(Mesh)

    # We center the data and compute the sample covariance matrix.
    Mesh_centered = Mesh - np.mean(Mesh, axis=0)
    cov_matrix = np.dot(Mesh_centered.T, Mesh_centered) / n_samples
    eigenvalues = pca.explained_variance_
    result = []
    for eigenvalue, eigenvector in zip(eigenvalues, pca.components_):
        print(np.dot(eigenvector.T, np.dot(cov_matrix, eigenvector)))
        print 'eigenvalue', (eigenvalue)
        print 'eigenvector', eigenvector
        result.append(eigenvector)
    result = np.array(result)
    return np.mean(Mesh, axis=0), result, eigenvalues


def save_landmark(file_path, landmark):
    with open(file_path, 'w') as file:
        if (landmark.shape[1] == 2):
            for i in range(0, landmark.shape[0]):
                write_str = '%f %f\n' % (landmark[i, 0], landmark[i, 1])
                file.write(write_str)

        elif landmark.shape[1] == 3:
            for i in range(0, landmark.shape[0]):
                write_str = '%f %f %f\n' % (landmark[i, 0], landmark[i, 1], landmark[i, 2])
                file.write(write_str)
        elif landmark.shape[1] == 1:
            for i in range(0, landmark.shape[0]):
                write_str = '%f\n' % (landmark[i, 0])
                file.write(write_str)


def save_int(file_path, landmark):
    with open(file_path, 'w') as file:
        if (landmark.shape[1] == 2):
            for i in range(0, landmark.shape[0]):
                write_str = '%d %d\n' % (landmark[i, 0], landmark[i, 1])
                file.write(write_str)

        elif landmark.shape[1] == 3:
            for i in range(0, landmark.shape[0]):
                write_str = '%d %d %d\n' % (landmark[i, 0], landmark[i, 1], landmark[i, 2])
                file.write(write_str)
        elif landmark.shape[1] == 1:
            for i in range(0, landmark.shape[0]):
                write_str = '%d\n' % (landmark[i, 0])
                file.write(write_str)


def read_int(file_path):
    result = []
    with open(file_path, 'r') as f:
        data = f.readlines()  # txt中所有字符串读入data
        for line in data:
            line = line.replace('\n', '')
            if line == '':
                break
            odom = line.split(' ')  # 将单个数据分隔开存好
            # print  odom
            numbers_float = map(int, odom)  # 转化为整数
            result.append(numbers_float)
    return np.array(result)


def read_landmark(file_path, split_char=','):
    result = []
    with open(file_path, 'r') as f:
        data = f.readlines()  # txt中所有字符串读入data
        for line in data:
            line = line.replace('\n', '')
            odom = line.split(split_char)  # 将单个数据分隔开存好
            numbers_float = map(float, odom)  # 转化为浮点数
            result.append(numbers_float)
    return np.array(result)


# 或者 a = numpy.loadtxt('odom.txt')
def add_vertex_faces(pre_v, pre_f, add_v, add_f):
    if pre_v.size == 0:
        new_v = add_v
        new_f = add_f
    else:
        new_v = np.vstack((pre_v, add_v))
        pre_num_vertices = pre_v.shape[0]
        add_f = add_f + pre_num_vertices
        new_f = np.vstack((pre_f, add_f))
    return new_v, new_f


def write_landmark_to_obj(file_path, landmark, size=1000):
    sphere_v = igl.eigen.MatrixXd()
    sphere_f = igl.eigen.MatrixXi()
    igl.readOBJ('sphere.obj', sphere_v,
                sphere_f)
    sphere_v = np.array(sphere_v)
    sphere_f = np.array(sphere_f)
    lmk_num = landmark.shape[0]
    sphere_v_move = np.array([])
    all_v = np.array([])
    all_f = np.array([])
    for i in range(0, lmk_num):
        sphere_v_move = size * sphere_v + landmark[i, :]
        all_v, all_f = add_vertex_faces(all_v, all_f, sphere_v_move, sphere_f)
    igl.writeOBJ(file_path, igl.eigen.MatrixXd(all_v),
                 igl.eigen.MatrixXi(all_f.astype('intc')))


def numpy_load(file_path):
    a = np.loadtxt(file_path)
    return a


def triangle_intersect(origin, dir, v1, v2, v3):
    origin = np.array(origin)
    dir = np.array(dir)
    v1 = np.array(v1)
    v2 = np.array(v2)
    v3 = np.array(v3)
    isintersect = False
    t = 0.0
    dir = dir / np.linalg.norm(dir)
    EPSILON = 10e-6
    result_point = []
    e1 = v2 - v1
    e2 = v3 - v1
    p = np.cross(dir, e2)
    tmp = p.dot(e1)
    if tmp > - EPSILON and tmp < EPSILON:
        return isintersect, result_point, t
    tmp = 1.0 / tmp
    s = origin - v1
    u = tmp * (s.dot(p))
    if (u < 0.0 or u > 1.0):
        return isintersect, result_point, t
    q = np.cross(s, e1)
    v = tmp * (dir.dot(q))
    if v < 0.0 or v > 1.0:
        return isintersect, result_point, t
    uv = u + v
    if uv > 1.0:
        return isintersect, result_point, t
    t = tmp * (e2.dot(q))
    if t < 0:
        isintersect = False
        return isintersect, result_point, t
    isintersect = True
    result_point = origin + t * dir
    return isintersect, result_point, t


def cast2d_to3d(point2d, v_mesh, f_mesh):
    result = []
    min_t = 2000
    for i in range(0, f_mesh.shape[0]):
        isintersect, intersect_point, t = triangle_intersect([point2d[0], point2d[1], 192], [0, 0, -1],
                                                             v_mesh[f_mesh[i, 0], :], v_mesh[f_mesh[i, 1], :],
                                                             v_mesh[f_mesh[i, 2], :])
        if isintersect == True:
            if t > 0:
                if t < min_t:
                    min_t = t
                    result = intersect_point
    if len(result) > 0:
        issucess = True
    else:
        issucess = False

    return issucess, result


# use trimesh

def cast2d_to3d_trimesh(mesh_path, point2d):
    import trimesh
    mesh = trimesh.load(mesh_path)
    point_num = point2d.shape[0]
    ray_origins = np.zeros((point_num, 3), dtype=np.float64)
    ray_directions = np.zeros((point_num, 3), dtype=np.float64)
    for i in range(0, point_num):
        ray_origins[i] = np.array([point2d[i, 0], point2d[i, 1], 192])
        ray_directions[i] = np.array([0, 0, -1])
    index_triangles, index_ray, result_locations = mesh.ray.intersects_id(ray_origins=ray_origins,
                                                                          ray_directions=ray_directions,

                                                                          multiple_hits=False, return_locations=True)
    return index_triangles, index_ray, result_locations


def scaleToOriCoodi_topleft(source, BB, scaled_size):
    if BB[2] != BB[3]:
        print 'BB has error'
    ori_size = BB[2]
    target = source / float(scaled_size) * ori_size
    target = target + np.array([BB[0], BB[2]])
    return target


def scaleToOriCoodi_bottomleft(source, BB, scaled_size):
    if BB[2] != BB[3]:
        print 'BB has error'
    ori_size = BB[2]
    target = source / float(scaled_size) * ori_size
    target[:, 0:2] = target[:, 0:2] + np.array([BB[0], BB[1]])
    return target


def write_image_and_featurepoint(image, feature_point, file_path):
    import cv2
    image = image.copy()
    count = 0
    for (x, y) in feature_point:
        if count < 8:
            cv2.circle(image, (int(x), int(y)), 1, (0, 0, 255), -1)
        elif count > 8 and count < 17:
            cv2.circle(image, (int(x), int(y)), 1, (0, 255, 0), -1)
        elif count in [27, 28, 29, 30, 33, 51, 62, 66, 57, 8]:
            cv2.circle(image, (int(x), int(y)), 1, (255, 255, 0), -1)
        elif count in [17, 18, 19, 20, 21, 36, 37, 38, 39, 40, 41]:
            cv2.circle(image, (int(x), int(y)), 1, (0, 255, 255), -1)
        elif count in [22, 23, 24, 25, 26, 42, 43, 44, 45, 46, 47]:
            cv2.circle(image, (int(x), int(y)), 1, (255, 0, 255), -1)
        elif count in [48, 49, 50, 60, 61, 67, 58, 59]:
            cv2.circle(image, (int(x), int(y)), 1, (255, 0, 0), -1)
        elif count in [52, 53, 54, 55, 56, 63, 64, 65]:
            cv2.circle(image, (int(x), int(y)), 1, (0, 255, 255), -1)
        elif count in [31, 32]:
            cv2.circle(image, (int(x), int(y)), 1, (0, 255, 0), -1)
        elif count in [34, 35]:
            cv2.circle(image, (int(x), int(y)), 1, (128, 0, 128), -1)
        count += 1
    cv2.imwrite(file_path, image)


def detect_68_landmark_dlib(image):  # 左上角为原点
    import dlib
    height = image.shape[0]
    width = image.shape[1]
    predictor_path = 'D:/mprojects/FaceProject/FaceProject/src/shape_predictor_68_face_landmarks.dat'
    predictor_path = 'shape_predictor_68_face_landmarks.dat'
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    dets = detector(image, 1)
    featurePointxy = np.zeros([68, 2])
    if len(dets) >= 0:
        shape = predictor(image, dets[0])
    for i in xrange(68):
        featurePointxy[i, 0] = shape.part(i).x
        featurePointxy[i, 1] = shape.part(i).y
    return featurePointxy


def readImage(path):
    # cv2.IMREAD_UNCHANGED 四通道
    import cv2
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    return image


def sym_point(front, up, normal, mean, point):
    m = np.vstack([front, up, normal])  # 投射矩阵
    front = front.reshape(front.size, 1)
    up = up.reshape(up.size, 1)
    normal = normal.reshape(normal.size, 1)
    m2 = np.concatenate((front, up, normal), axis=1)
    project_coff = np.dot(m, (point - mean))  # 使用 .dot才是矩阵乘法，否则是对应相乘
    project_coff[2] = -project_coff[2]  # front,up 平面对称
    new_point = m2.dot(project_coff) + mean
    return new_point


def corr_point(front, up, normal, mean, point):
    m = np.vstack([front, up, normal])  # 投射矩阵
    front = front.reshape(front.size, 1)
    up = up.reshape(up.size, 1)
    normal = normal.reshape(normal.size, 1)
    m2 = np.concatenate((front, up, normal), axis=1)
    #    m2 = np.hstack([front.T,up.T,normal.T]) #基矩阵
    project_coff = np.dot(m, (point - mean))  # 使用 .dot才是矩阵乘法，否则是对应相乘
    project_coff[2] = 0  # front,up 平面对称
    new_point = m2.dot(project_coff) + mean
    return new_point


def corr_landmark_tofit_data(front, up, normal, mean, mesh, landmark):
    if (len(landmark.shape) == 1):
        landmark = landmark.reshape(1, landmark.size)
    m = np.vstack([front, up, normal])  # 投射矩阵
    centered_mesh = mesh - mean
    project_coff = np.dot(m, (centered_mesh.T))  # 使用 .dot才是矩阵乘法，否则是对应相乘
    project_coff = project_coff.T

    up_min = min(project_coff[:, 1])
    up_max = max(project_coff[:, 1])
    grid_num = 200
    depth_buf = np.zeros((grid_num, 1), np.float64)
    step = (up_max - up_min) / grid_num
    for i in range(0, project_coff.shape[0]):
        # 去掉偏离中心面的点
        if abs(project_coff[i, 2]) > 3:
            continue
        curr_depth = project_coff[i, 0]
        grid_id = int((project_coff[i, 1] - up_min) / step)
        if grid_id < 0:
            grid_id = 0
        if grid_id > grid_num - 1:
            grid_id = grid_num - 1
        if depth_buf[grid_id] < curr_depth:
            depth_buf[grid_id] = curr_depth
    #    print  depth_buf

    landmark = landmark - mean
    project_landmark = np.dot(m, (landmark.T))
    project_landmark = project_landmark.T
    for i in range(0, project_landmark.shape[0]):
        grid_id = int((project_landmark[i, 1] - up_min) / step)
        if grid_id < 0:
            grid_id = 0
        if grid_id > grid_num - 1:
            grid_id = grid_num - 1
        project_landmark[i, 0] = depth_buf[grid_id]
    front = front.reshape(front.size, 1)
    up = up.reshape(up.size, 1)
    normal = normal.reshape(normal.size, 1)
    m2 = np.concatenate((front, up, normal), axis=1)
    project_landmark = m2.dot(project_landmark.T)
    project_landmark = project_landmark.T + mean

    buf_vertex = []
    for i in range(0, 200):
        buf_vertex.append([depth_buf[i], up_min + step * i, 0])
    buf_vertex = np.array(buf_vertex)
    buf_vertex = m2.dot(buf_vertex.T)
    buf_vertex = buf_vertex.T + mean

    return project_landmark, buf_vertex


def sym_plane(front, up, normal, mean, scale):
    vertex = np.zeros((9, 3))
    vertex[0, :] = mean + front * scale + up * scale
    vertex[1, :] = mean + front * scale
    vertex[2, :] = mean + [0, 0, 0]
    vertex[3, :] = mean + up * scale
    vertex[4, :] = mean + -front * scale
    vertex[5, :] = mean + -front * scale + up * scale
    vertex[6, :] = mean + front * scale - up * scale
    vertex[7, :] = mean + - up * scale
    vertex[8, :] = mean + -front * scale - up * scale
    vertex = np.array(vertex)
    f = [[0, 1, 2], [0, 2, 3], [3, 2, 4], [3, 4, 5], [1, 6, 7], [1, 7, 2], [2, 7, 8], [2, 8, 4]]
    f = np.array(f)
    return vertex, f


def cast_coeff(front, point):
    coeff = front.dot(point)
    return coeff


def determinant(v1, v2, v3, v4):
    result = v1.dot(v3) - v2.dot(v4)


def convert_para_tex(para_objmesh, paraTexMat):
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
    texPC = target['texPC'][0, 0]
    texMU = target['texMU'][0, 0]
    texEV = target['texEV'][0, 0]

    for i_pc in range(0, 1):
        '''
        coff = np.zeros((199,1))
        coff[i_pc] = 5e3
        cur_tex = texMU+np.dot(texPC,coff)
        cur_tex = cur_tex.reshape(cur_tex.size/3,3)
        output_path = './texpc/' + 'pc_' + str(i_pc) + '_coff_' + str(float(coff[i_pc])) + '.obj'
        write_full_obj(v,f,n,n_f,t,t_f,cur_tex/255.0,output_path)
        '''
        coff = np.zeros((199, 1))
        coff[i_pc] = 4.1031792e3 * 10
        print coff.shape
        coff.reshape(coff.size, 1)
        print coff.shape
        cur_tex = texMU + np.dot(texPC, coff)
        for i in range(0, cur_tex.size):
            if cur_tex[i] < 0.0:
                cur_tex[i] = 0.0
            if cur_tex[i] > 255.0:
                cur_tex[i] = 255.0
            cur_tex[i] /= 255.0

        cur_tex = cur_tex.reshape(cur_tex.size / 3, 3)

        output_path = './texpc/' + 'pc_' + str(i_pc) + '_coff_' + str(float(coff[i_pc])) + '.obj'
        write_full_obj(v, f, n, n_f, t, t_f, cur_tex, output_path)

    '''
        igl.writeOBJ(output_path,
                    igl.eigen.MatrixXd(v.astype('float64')),igl.eigen.MatrixXi(f.astype('intc')),
                    igl.eigen.MatrixXd(n.astype('float64')),igl.eigen.MatrixXi(n_f.astype('intc')),
                    igl.eigen.MatrixXd(t.astype('float64')),t_f.astype('intc'))

    '''


Const_Image_Format = [".jpg", ".jpeg", ".bmp", ".png", ".obj", ".ppm", ".data"]


class FileFilt:

    def __init__(self):
        self.fileList = [""]
        self.counter = 0
        pass

    def FindFile(self, dirr, filtrate=1):
        global Const_Image_Format
        for s in os.listdir(dirr):
            newDir = os.path.join(dirr, s)
            if os.path.isfile(newDir):
                if filtrate:
                    if newDir and (os.path.splitext(newDir)[1] in Const_Image_Format):
                        self.fileList.append(newDir)
                        self.counter += 1
                else:
                    self.fileList.append(newDir)
                    self.counter += 1


'''
use example
if __name__ == "__main__":
        b = FileFilt()
        b.FindFile(dirr = "D:\Python27\user\dapei-imgs")
        print(b.counter)
        for k in b.fileList:
            print k
'''


def process_maya_select_vtx(maya_path):
    result = []
    with open(maya_path, 'r') as f:
        data = f.readlines()  # txt中所有字符串读入data
        for line in data:
            odom = line.split(' ')  # 将单个数据分隔开存好
            for part in odom:
                left_quo = part.find('[')
                right_quo = part.find(']')
                if left_quo > -1 and right_quo > -1:
                    extract = part[left_quo + 1:right_quo]
                    maohao_pos = extract.find(':')
                    if maohao_pos > -1:
                        left_pos = extract[0:maohao_pos]
                        right_pos = extract[maohao_pos + 1:len(extract)]
                        left_pos = int(left_pos)
                        right_pos = int(right_pos)
                        for pos in range(left_pos, right_pos + 1):
                            result.append(pos)
                    else:
                        pos = int(extract)
                        result.append(pos)
    return np.array(result)


def cac_normal(v_np, f_np):
    V_igl = igl.eigen.MatrixXd(v_np.astype('float64'))
    F_igl = igl.eigen.MatrixXi(f_np.astype('intc'))
    VertexNormal = igl.eigen.MatrixXd()
    igl.per_vertex_normals(V_igl, F_igl, igl.PER_VERTEX_NORMALS_WEIGHTING_TYPE_AREA, VertexNormal)
    return np.array(VertexNormal)


def read_igl_obj(para_objmesh):
    v = igl.eigen.MatrixXd()
    f = igl.eigen.MatrixXi()
    n = igl.eigen.MatrixXd()
    n_f = igl.eigen.MatrixXi()
    t = igl.eigen.MatrixXd()
    t_f = igl.eigen.MatrixXi()
    igl.readOBJ(para_objmesh, v, t, n, f, t_f, n_f)
    v = np.array(v)
    v = v[:, 0:3]
    t = np.array(t)
    n = np.array(n)
    f = np.array(f)
    t_f = np.array(t_f)
    n_f = np.array(n_f)
    return v, f, t, t_f, n, n_f


# get the longet order loop
def boudaray_loop(F):
    F = igl.eigen.MatrixXi(F.astype('intc'))
    F1 = igl.eigen.MatrixXi()
    igl.boundary_loop(F, F1)
    F1 = np.array(F1)
    return F1


def map_vertices_to_circle(V, bnd):
    UV = igl.eigen.MatrixXd()
    UV1 = igl.eigen.MatrixXd()
    a = igl.eigen.MatrixXi()
    UV1 = igl.eigen.MatrixXd(V.astype('float64'))
    igl_bnd = igl.eigen.MatrixXi(bnd.astype('intc'))
    igl.map_vertices_to_circle(UV1, igl_bnd, UV)
    UV = np.array(UV)
    return UV


def Harmonic(V, F, b, bc, k=1):
    """
    V : N*3 Vertex positon
    F : M*3 face list
    b: boundary indices in V
    BC : target boudary values ,2D, K*2
    k power of harmonic operation : 1 harmonic ,2 biharmoic,3
    :param V:
    :param F:
    :param b:
    :param bc:
    :param k:
    :return:
    """
    W = igl.eigen.MatrixXd()
    igl.harmonic(igl.eigen.MatrixXd(V.astype('float64')), igl.eigen.MatrixXi(F.astype('intc')),
                 igl.eigen.MatrixXi(b.astype('intc')), igl.eigen.MatrixXd(bc.astype('float64')), int(k), W)
    W = np.array(W)
    return W


class ARAP_DATA:
    """
    0 ARAP_ENERGY_TYPE_DEFAULT
    1 ARAP_ENERGY_TYPE_ELEMENTS
    2 ARAP_ENERGY_TYPE_SPOKES_AND_RIMS
    3 ARAP_ENERGY_TYPE_SPOKES
    """

    def __init__(self, energy_type=0):
        self.data = igl.ARAPData()
        self.data.with_dynamics = True
        self.data.max_iter = 100
        self.data.ym = 1
        energy_type = int(energy_type)
        if energy_type == 0:
            self.data.energy = igl.ARAP_ENERGY_TYPE_DEFAULT
        elif energy_type == 1:
            self.data.energy = igl.ARAP_ENERGY_TYPE_ELEMENTS
        elif energy_type == 2:
            self.data.energy = igl.ARAP_ENERGY_TYPE_SPOKES_AND_RIMS
        elif energy_type == 3:
            self.data.energy = igl.ARAP_ENERGY_TYPE_SPOKES
        else:
            self.energy = igl.ARAP_ENERGY_TYPE_DEFAULT


def arap_precomputation(V, F, dim, bt, arap_data):
    igl.arap_precomputation(igl.eigen.MatrixXd(V.astype('float64')), igl.eigen.MatrixXi(F.astype('intc')),
                            dim, igl.eigen.MatrixXi(bt.astype('intc')), arap_data)
    return arap_data


def arap_solve(bc, arap_data, initial_uv):
    u_uv = igl.eigen.MatrixXd(initial_uv.astype('float64'))
    igl.arap_solve(igl.eigen.MatrixXd(bc.astype('float64')), arap_data,
                   u_uv)
    u_uv = np.array(u_uv)
    return u_uv


# 我们以[0,1] 作为起点轴，顺时针旋转
def get_rad_from2d_dir(dir):
    # import math.pi as pi
    # import math.atan as atan
    if np.linalg.norm(dir) < 0.0001:
        print 'dir wrong'
    dir /= np.linalg.norm(dir)
    dir = dir[0:2]  # cast 2d
    if np.linalg.norm(dir) < 0.0001:
        return [255, 255, 255]
    else:
        dir /= np.linalg.norm(dir)
    x = dir[0]
    y = dir[1]
    # 我们以[0,1] 作为起点轴，顺时针旋转
    if abs(y) < 0.0001:
        if x > 0:
            theta = math.pi / 2
        else:
            theta = 3 * math.pi / 2
    else:
        theta = math.atan(x / y)
        if theta >= 0:
            if y > 0 and x >= 0:
                pass
            else:  # y<0 and x <0
                theta += math.pi
        else:
            if y < 0 and x >= 0:
                theta += math.pi
            else:  # y>0 ,x<0
                theta += 2 * math.pi
    return theta


# color = color_fun(rad)
def genertate_dir_color(img_width, img_height, path, color_fun):
    orientation_img = np.zeros((img_height, img_width, 3), np.uint8)  # BGRA
    center_y = int(img_height / 2)
    center_x = int(img_width / 2)
    R_max = min([center_x, center_y])
    R_max = int(R_max * 3.0 / 4.0)
    R_min = int(R_max / 2)
    for y in range(0, img_height):
        for x in range(0, img_width):
            y_revert = img_height - 1 - y
            x_2 = (x - center_x) * (x - center_x)
            y_2 = (y_revert - center_y) * (y_revert - center_y)
            if x_2 + y_2 <= R_max * R_max and x_2 + y_2 >= R_min * R_min:
                rad = get_rad_from2d_dir([x - center_x, y_revert - center_y])
                color = color_fun(rad)
                # print y,x,rad,color
                orientation_img[y, x, 0:3] = color[::-1]
                # orientation_img[y, x, 3] = 255
            else:
                orientation_img[y, x, :] = [255, 255, 255]

    import cv2
    cv2.imwrite(path, orientation_img)


def read_strand_txt_file(file_path):
    hair = []
    # strand_num = 0
    # strand_len = 0
    count = 0
    try:
        with open(file_path, 'r') as f:
            data = f.readlines()  # txt中所有字符串读入data

            for line in data:
                line = line.replace("\n", "")
                odom = line.split(' ')  # 将单个数据分隔开存好
                if len(odom) > 0:
                    # print 'read_strand_txt_file',count,len(odom)
                    if 0 == count:
                        pass
                        # print 'strand num',int(odom[0])
                    if count % 2 == 1:
                        pass
                        # print  'strand len',int(odom[0])
                    elif count > 0:
                        strand = []
                        for i in range(0, len(odom) / 2):
                            pixel = [float(odom[2 * i]), float(odom[2 * i + 1])]
                            strand.append(pixel)
                        hair.append(strand)
                    count += 1
            return np.array(hair)

            # data = f.readlines()  # txt中所有字符串读入data
            # for line in data:
            #     odom = line.split(' ')  # 将单个数据分隔开存好
            #     if len(odom) > 0:
            #         pass


    except IOError:
        return np.array(hair)


# 以左下角为原点 ,input_img 应是左下角坐标系，返回的img也是按照做下角坐标系
def rescale_imge_with_bbox(input_img, bbox_2d, color=[0, 0, 0]):
    left_bottom_x = int(bbox_2d[0])  # int(bbox_2d.min[0])
    left_bottom_y = int(bbox_2d[1])  # int(bbox_2d.min[1])
    right_top_x = int(bbox_2d[2])  # int(bbox_2d.max[0])
    right_top_y = int(bbox_2d[3])  # int(bbox_2d.max[1])
    # 包括包围盒边界的像素
    new_width = right_top_x - left_bottom_x + 1
    new_height = right_top_y - left_bottom_y + 1
    new_img = np.zeros((new_height, new_width, input_img.shape[2]), np.uint8)
    new_img[:, :, :] = color
    for j in range(0, input_img.shape[0]):
        for i in range(0, input_img.shape[1]):
            if j < left_bottom_y or j > right_top_y or i < left_bottom_x or i > right_top_x:
                continue
            # 说明这里 的   j >= left_bottom_y and j < = right_top_y and  i >=  left_bottom_x and i <= right_top_x
            new_index_y = j - left_bottom_y
            new_index_x = i - left_bottom_x
            new_img[new_index_y, new_index_x, :] = input_img[j, i, :]

    return new_img


def get_matrix_from_euler(euler):
    """ 获得两组对应点数的二维点间的最佳 刚性变换和 放缩单值
    scale*rotate3d*landmark_2d_source + translate_2d
    :param euler:
    :return:
    """
    import chumpy as ch
    a = euler[0]
    b = euler[1]
    c = euler[2]
    Rx = ch.vstack(([1, 0, 0],
                    ch.concatenate((0, ch.cos(a), -ch.sin(a))),
                    ch.concatenate((0, ch.sin(a), ch.cos(a)))))
    Ry = ch.vstack((ch.concatenate((ch.cos(b), 0, ch.sin(b))),
                    [0, 1, 0],
                    ch.concatenate((-ch.sin(b), 0, ch.cos(b)))))
    Rz = ch.vstack((ch.concatenate((ch.cos(c), -ch.sin(c), 0)),
                    ch.concatenate((ch.sin(c), ch.cos(c), 0)),
                    [0, 0, 1]))
    Rot2 = ch.dot(Rz, ch.dot(Ry, Rx))
    return Rot2


def get_opt_transform_2d(landmark_3d_source, landmark_2d_target):
    import chumpy as ch
    landmark_3d_source = ch.array(landmark_3d_source)
    landmark_2d_target = ch.array(landmark_2d_target)

    def landmark_error(scale, translate_2d, euler):
        Rot2 = get_matrix_from_euler(euler)

        reshape_landmark = scale * ch.dot(Rot2, landmark_3d_source.T)

        landmark_distance = 1.0 / scale * (reshape_landmark.T[:, 0:2] + translate_2d - landmark_2d_target)
        # landmark_distance = reshape_landmark3[:, 0:2] + translate_2d - landmark_2d_target
        return landmark_distance

    scale = ch.array([1])
    translate_2d = ch.array([0, 0])
    euler = ch.array([0, 0, 0])  # xtheata, y theata ,z theta
    landmark_distance = landmark_error(scale=scale, translate_2d=translate_2d, euler=euler)
    objectives = {}
    objectives.update({'landmark_distance': landmark_distance
                       })

    def on_step(_):
        pass

    import scipy.sparse as sp
    from time import time
    opt_options_20 = {}
    opt_options_20['disp'] = 1
    opt_options_20['delta_0'] = 0.1
    opt_options_20['e_3'] = 1e-4
    opt_options_20['maxiter'] = 50
    sparse_solver_20 = lambda A, x: sp.linalg.cg(A, x, maxiter=opt_options_20['maxiter'])[0]
    opt_options_20['sparse_solver'] = sparse_solver_20

    def print_para_resut(step, timer_end, timer_start):
        print step
        print "in %f sec\n" % (timer_end - timer_start)
        print "euler"
        print euler
        print "scale"
        print scale
        print "translate_2d"
        print translate_2d

    # 发现一个问题，如果把 translation ,scale ,rotate 放在一起优化，存在局部最小问题效果不佳，因此考虑先优化 scale,translate,再优化 translate ,rotate
    timer_start = time()
    # 先优化 scale, translate
    ch.minimize(fun=objectives,
                x0=[scale, translate_2d],
                method='dogleg',
                callback=on_step,
                options=opt_options_20)

    timer_end = time()
    print_para_resut('get_opt_transform', timer_start, timer_end)

    # 再优化 , translate ，rotate
    objectives = {}
    objectives.update({'landmark_distance': landmark_distance, 'euler': euler
                       })
    ch.minimize(fun=objectives,
                x0=[translate_2d, euler],
                method='dogleg',
                callback=on_step,
                options=opt_options_20)

    timer_end = time()
    print_para_resut('get_opt_transform', timer_start, timer_end)

    return scale.r, get_matrix_from_euler(euler).r, translate_2d.r


def get_opt_transform_3d(landmark_3d_source, landmark_3d_target):
    """ 获得两组对应点数的三维点间的最佳 刚性变换和 放缩单值
    scale*rotate3d*landmark_2d_source + translate_2d
    :param landmark_3d_source:
    :param landmark_3d_target:
    :return:
    """
    print("get_opt_transform_3d ")
    import chumpy as ch
    landmark_3d_source = ch.array(landmark_3d_source)

    def landmark_error(scale, translate_3d, euler):
        Rot2 = get_matrix_from_euler(euler)
        reshape_landmark = scale * ch.dot(Rot2, landmark_3d_source.T)
        landmark_distance = reshape_landmark.T[:, :] + translate_3d - landmark_3d_target
        return landmark_distance

    scale = ch.array([1])
    translate_3d = ch.array([0, 0, 0])
    euler = ch.array([0, 0, 0])  # xtheata, y theata ,z theta
    landmark_distance = landmark_error(scale=scale, translate_3d=translate_3d, euler=euler)
    objectives = {}
    objectives.update({'landmark_distance': landmark_distance, 'euler': euler
                       })

    def on_step(_):
        pass

    import scipy.sparse as sp
    from time import time
    opt_options_20 = {}
    opt_options_20['disp'] = 1
    opt_options_20['delta_0'] = 0.1
    opt_options_20['e_3'] = 1e-4
    opt_options_20['maxiter'] = 20
    sparse_solver_20 = lambda A, x: sp.linalg.cg(A, x, maxiter=opt_options_20['maxiter'])[0]
    opt_options_20['sparse_solver'] = sparse_solver_20

    def print_para_resut(step, timer_end, timer_start):
        print step
        print "in %f sec\n" % (timer_end - timer_start)
        print "euler"
        print euler
        print "scale"
        print scale
        print "translate_3d"
        print translate_3d

    timer_start = time()
    ch.minimize(fun=objectives,
                x0=[scale, translate_3d, euler],
                method='dogleg',
                callback=on_step,
                options=opt_options_20)

    timer_end = time()
    print_para_resut('get_opt_transform', timer_start, timer_end)
    return scale.r, get_matrix_from_euler(euler).r, translate_3d.r


def write_frame_obj(path):
    model_path = './models/female_model.pkl'  # change to 'female_model.pkl' or 'generic_model.pkl', if needed
    from smpl_webuser.serialization import load_model

    model = load_model(model_path)
    write_simple_obj(model.r, model.f, path)
    # write_simple_obj(model.r,model.f,'E:/workspace/dataset/hairstyles/frame_female_init.obj')


def mesh_points_by_barycentric_coordinates(mesh_verts, mesh_faces, lmk_face_idx, lmk_b_coords):
    """ function: evaluation 3d points given mesh and landmark embedding
    """
    dif1 = np.vstack([(mesh_verts[mesh_faces[lmk_face_idx], 0] * lmk_b_coords).sum(axis=1),
                      (mesh_verts[mesh_faces[lmk_face_idx], 1] * lmk_b_coords).sum(axis=1),
                      (mesh_verts[mesh_faces[lmk_face_idx], 2] * lmk_b_coords).sum(axis=1)]).T
    return dif1


def get_vertex_normal(V_np, F_np):
    V_igl = igl.eigen.MatrixXd(V_np.astype('float64'))
    F_igl = igl.eigen.MatrixXi(F_np.astype('intc'))
    VertexNormal = igl.eigen.MatrixXd()
    igl.per_vertex_normals(V_igl, F_igl, igl.PER_VERTEX_NORMALS_WEIGHTING_TYPE_AREA, VertexNormal)
    VertexNormal = np.array(VertexNormal)
    return VertexNormal


# 均为左下角作为原点
def sample_color_from_img(vertex, vertex_normal, image, invaid_img=None):
    color = np.zeros((vertex.shape[0], 3), np.uint8)
    if invaid_img is None or invaid_img.any() == None:
        invaid_img = np.ones((image.shape[0], image.shape[1]), dtype=np.bool)
        invaid_img[:, :] = True
    for i in range(0, vertex.shape[0]):
        x = vertex[i, 0]
        y = vertex[i, 1]
        normal = vertex_normal[i]
        dir = np.array([0, 0, 1])
        if np.dot(normal, dir) < 0.01:
            continue
        x = int(x)
        y = int(y)

        if x < 0:
            x = 0
        if y < 0:
            y = 0
        if x > image.shape[1] - 1:
            x = image.shape[1] - 1
        if y > image.shape[0] - 1:
            y = image.shape[0] - 1
        if not invaid_img[y, x]:
            continue
        bgr = image[y, x]
        rgb = bgr[::-1]
        color[i, :] = rgb
    return color


# 均为左下角作为原点
# 假设 image 是mxn
def sample_intensity_from_img(vertex, vertex_normal, image):
    color = np.zeros((vertex.shape[0], 1), np.uint8)
    for i in range(0, vertex.shape[0]):
        x = vertex[i, 0]
        y = vertex[i, 1]
        normal = vertex_normal[i]
        dir = np.array([0, 0, 1])
        # if np.dot( normal,dir)<0.01:
        #     continue
        x = int(round(x))
        y = int(round(y))
        if x < 0:
            x = 0
        if y < 0:
            y = 0
        if x > image.shape[1] - 1:
            x = image.shape[1] - 1
        if y > image.shape[0] - 1:
            y = image.shape[0] - 1
        intensity = image[y, x]
        color[i, :] = intensity
    return color


def convert_seg_to_binary(img, front_color):
    convert_img = np.zeros((img.shape[0], img.shape[1], 1), np.uint8)
    for j in range(0, img.shape[0]):
        for i in range(0, img.shape[1]):
            if img[j, i, 0] == front_color[0] and img[j, i, 1] == front_color[1] and img[j, i, 2] == front_color[2]:
                convert_img[j, i, :] = 255
    return convert_img


def smooth_seg_image(input_path, out_put_path, front_color=[255, 255, 255]):
    import cv2
    INPUT = cv2.imread(input_path, cv2.IMREAD_COLOR)
    INPUT = convert_seg_to_binary(INPUT, front_color)
    if 0:
        MASK = np.array(INPUT / 255.0, dtype='float32')

        MASK = cv2.GaussianBlur(MASK, (5, 5), 11)
        BG = np.ones([INPUT.shape[0], INPUT.shape[1], 1], dtype='uint8') * 255

        OUT_F = np.ones([INPUT.shape[0], INPUT.shape[1], 1], dtype='uint8')

        for r in range(INPUT.shape[0]):
            for c in range(INPUT.shape[1]):
                OUT_F[r][c] = int(BG[r][c] * (MASK[r][c]) + INPUT[r][c] * (1 - MASK[r][c]))

        cv2.imwrite(out_put_path, OUT_F)
    else:

        ret, INPUT = cv2.threshold(INPUT, 125, 255, cv2.THRESH_BINARY)
        blurredImage = cv2.pyrUp(INPUT)
        for i in range(0, 20):
            blurredImage = cv2.medianBlur(blurredImage, 7)
        blurredImage = cv2.pyrDown(blurredImage, blurredImage)
        ret, blurredImage = cv2.threshold(blurredImage, 200, 255, cv2.THRESH_BINARY)
        # blurredImage =get_binaray_img_boundary(blurredImage)
        cv2.imwrite(out_put_path, blurredImage)


def get_binaray_img_boundary(binary_img, threshold=1):
    coutour_img = np.zeros((binary_img.shape[0], binary_img.shape[1], 3), np.uint8)
    coutour_img[:, :, :] = [255, 255, 255]

    boundary_pixel = []  # x,y格式
    for j in range(0, binary_img.shape[0]):
        for i in range(0, binary_img.shape[1]):
            is_boundary = False
            if binary_img[j, i] < threshold:
                continue
            if j > 0:
                if binary_img[j - 1, i] < threshold:
                    is_boundary = True
            if j < binary_img.shape[0] - 1:
                if binary_img[j + 1, i] < threshold:
                    is_boundary = True
            if i > 0:
                if binary_img[j, i - 1] < threshold:
                    is_boundary = True
            if i < binary_img.shape[1] - 1:
                if binary_img[j, i + 1] < threshold:
                    is_boundary = True
            if is_boundary:
                boundary_pixel.append([i, j])  # x,y 格式
    return boundary_pixel


def test_frame_mesh_landmark(input_mesh, out_put_landmark, size=1000):
    v_frame_init, f_frame_init, t_frame_init, t_f_frame_init, n_frame_init, n_f_frame_init = read_igl_obj(input_mesh)
    # landmark embedding
    lmk_emb_path = './data/lmk_embedding_intraface_to_flame.pkl'
    from fitting.util import mesh_points_by_barycentric_coordinates
    from fitting.landmarks import load_embedding
    lmk_face_idx, lmk_b_coords = load_embedding(lmk_emb_path)
    from fitting.util import mesh_points_by_barycentric_coordinates, get_vertex_normal, sample_color_from_img
    frame_landmark_3d = mesh_points_by_barycentric_coordinates(v_frame_init, f_frame_init, lmk_face_idx, lmk_b_coords)
    write_landmark_to_obj(out_put_landmark, frame_landmark_3d, size)


def build_adjacency(v, face):
    adjacency = {}
    for i in range(0, face.shape[0]):
        i0 = face[i, 0]
        i1 = face[i, 1]
        i2 = face[i, 2]
        if i0 in adjacency:
            pass
            # continue
            adjacency[i0].append(i1)
            adjacency[i0].append(i2)
        else:
            adjacency[i0] = []
            adjacency[i0].append(i1)
            adjacency[i0].append(i2)

        if i1 in adjacency:
            pass
            adjacency[i1].append(i2)
            adjacency[i1].append(i0)
        else:
            adjacency[i1] = []
            adjacency[i1].append(i2)
            adjacency[i1].append(i0)
        if i2 in adjacency:
            pass
            # continue
            adjacency[i2].append(i0)
            adjacency[i2].append(i1)
        else:
            adjacency[i2] = []
            adjacency[i2].append(i0)
            adjacency[i2].append(i1)
    # for k in adjacency.keys():
    #     adjacency[k] = list(set(adjacency[k]))
    return adjacency


# 假设 depth_img 二维
def convert_img_2_mseh_new(img, img_bool, depth_img=np.array([])):
    # 假设输入是bool 类型
    height = img.shape[0]
    width = img.shape[1]
    if depth_img.size == 0:
        depth_img = np.zeros((height, width))
    if img.shape[0] != img_bool.shape[0] or img.shape[1] != img_bool.shape[1]:
        print 'img shape not consitent'
    valid_pixels = np.zeros((height * width), dtype=bool)
    valid_pixels[:] = False
    face = []
    vertex = []  # np.zeros((height*width,3))
    for j in range(0, height):
        for i in range(0, width):
            if img_bool[j, i]:
                valid_count = 0
                if i > 0:
                    left_pixel = img_bool[j, i - 1]
                    if left_pixel:
                        valid_count += 1
                if i < width - 1:
                    right_pixel = img_bool[j, i + 1]
                    if right_pixel:
                        valid_count += 1
                if j > 0:
                    top_pixel = img_bool[j - 1, i]
                    if top_pixel:
                        valid_count += 1
                if j < height - 1:
                    down_pixel = img_bool[j + 1, i]
                    if down_pixel:
                        valid_count += 1
                # 如果valid_count等于0, 说明它的领域都为False
                if valid_count == 0:
                    pass
                else:
                    valid_pixels[j * width + i] = True

    for j in range(0, height):
        for i in range(0, width):
            if i == width - 1:
                cur_vertex = [i, height - 1 - j, depth_img[j, i]]
                vertex.append(cur_vertex)
            elif j == height - 1:
                cur_vertex = [i, height - 1 - j, depth_img[j, i]]
                vertex.append(cur_vertex)
            else:
                cur_index = j * width + i
                next_index = (j + 1) * width + i
                next2_index = (j + 1) * width + i + 1
                next3_index = j * width + i + 1
                face.append([cur_index, next_index, next2_index])
                face.append([cur_index, next2_index, next3_index])
                cur_vertex = [i, height - 1 - j, depth_img[j, i]]  # 深度认为是0
                vertex.append(cur_vertex)
                # next_vertex = [i, height - 1 - (j + 1), 0]
                # next_vertex2 = [i + 1, height - 1 - (j + 1), 0]
                # next_vertex3 = [i + 1, height - 1 - j, 0]

    # 去除invalid
    count = 0
    new_corr = np.zeros(height * width, np.uint)
    new_vertex = []
    new_color = []
    for j in range(0, height):
        for i in range(0, width):
            cur_index = j * width + i
            if valid_pixels[cur_index]:
                new_corr[cur_index] = count
                count += 1
                new_vertex.append(vertex[cur_index])
                if img[j, i].size == 1:
                    # print img[j,i]
                    new_color.append([img[j, i], img[j, i], img[j, i]])  # 转换为rgb格式
                else:
                    new_color.append(img[j, i, ::-1])  # 转换为rgb格式

    new_face = []
    for i in range(0, len(face)):
        cur_face = face[i]
        for i in range(0, 3):
            if valid_pixels[cur_face[0]] and valid_pixels[cur_face[1]] and valid_pixels[cur_face[2]]:
                new_face.append([new_corr[cur_face[0]], new_corr[cur_face[1]], new_corr[cur_face[2]]])
                pass
            else:
                pass

    return np.array(new_vertex), np.array(new_face), np.array(new_color)


def mesh_loop(mesh_v, mesh_f, mesh_n, mesh_n_f, mesh_tex, mesh_tex_f, vertexcolor, filepath, loopnum):
    import sys
    from configs.config import meshlab_python_path
    sys.path.insert(0, meshlab_python_path)
    import meshlab_python
    vertexcolor = vertexcolor.astype(np.int)
    result = meshlab_python.mesh_loop(
        filepath,
        mesh_v.tolist(), mesh_f.tolist(), mesh_n.tolist(), mesh_n_f.tolist(), mesh_tex.tolist(), mesh_tex_f.tolist(),
        vertexcolor.tolist(),
        int(loopnum))
    pass


def get_vertex_to_obj(landmark, size=1000):
    import os
    sphere_v = igl.eigen.MatrixXd()
    sphere_f = igl.eigen.MatrixXi()
    if os.path.exists('sphere.obj'):
        igl.readOBJ('sphere.obj', sphere_v,
                    sphere_f)
    else:
        raise RuntimeError('sphere path not right')
    sphere_v = np.array(sphere_v)
    sphere_f = np.array(sphere_f)
    lmk_num = landmark.shape[0]
    sphere_v_move = np.array([])
    all_v = np.array([])
    all_f = np.array([])
    for i in range(0, lmk_num):
        sphere_v_move = size * sphere_v + landmark[i, :]
        all_v, all_f = add_vertex_faces(all_v, all_f, sphere_v_move, sphere_f)
    return all_v, all_f


def laplacian_smooth(file_in_obj, file_out_obj, iteration=1, boudary=False, cotangent_weight=False, selected=False):
    # 注意把meshlabserver.exe 的路径 加入环境变量PATH中
    import meshlabxml as mlx
    smooth_script = mlx.FilterScript(file_in=file_in_obj, file_out=file_out_obj, ml_version='2016.12')
    mlx.remesh.smooth.laplacian(smooth_script, iteration, boudary, cotangent_weight, selected)
    smooth_script.run_script()


def convert_to_pure_triangle_mesh(file_in_obj, file_out_obj):
    import meshlabxml as mlx
    convert_to_triangle_script = mlx.FilterScript(file_in=file_in_obj, file_out=file_out_obj, ml_version='2016.12')


def moveFileto(sourceDir, targetDir):
    import shutil
    shutil.copy(sourceDir, targetDir)


'''
input_mesh_path1  被比较的
input_mesh_path2 以这个为基础输出网格

'''


def mesh_error_compare(input_mesh_path1, input_mesh_path2, output_mesh_path):
    subdived_mesh, subdived_mesh_f, t_frame_aligned, t_f_frame_aligned, n_frame_aligned, n_f_frame_aligned = read_igl_obj(
        input_mesh_path1)
    subdived_mesh2, subdived_mesh_f2, t_frame_aligned2, t_f_frame_aligned2, n_frame_aligned2, n_f_frame_aligned2 = read_igl_obj(
        input_mesh_path2)

    from fitting.measure import mesh2mesh, distance2color
    from sklearn.neighbors import NearestNeighbors
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(subdived_mesh)
    print 'neigh.fit'
    distances, indices = neigh.kneighbors(subdived_mesh2, return_distance=True)
    distance = mesh2mesh(subdived_mesh2, subdived_mesh[indices[:, 0]])
    vmax = 0.3
    color_3d = distance2color(dist=distance, vmin=0, vmax=vmax, cmap_name='jet')
    write_full_obj(subdived_mesh2, subdived_mesh_f2, n_frame_aligned2, n_f_frame_aligned2, t_frame_aligned2,
                   t_f_frame_aligned2,
                   color_3d,
                   output_mesh_path)


from fitting.measure import signdistance


def mesh_error_compare2(input_mesh_path1, input_mesh_path2, output_mesh_path):
    """

    :param input_mesh_path1: 被比较的
    :param input_mesh_path2: 以这个为基础输出网格
    :param output_mesh_path:
    :return:
    """
    subdived_mesh, subdived_mesh_f, t_frame_aligned, t_f_frame_aligned, n_frame_aligned, n_f_frame_aligned = read_igl_obj(
        input_mesh_path1)
    subdived_mesh2, subdived_mesh_f2, t_frame_aligned2, t_f_frame_aligned2, n_frame_aligned2, n_f_frame_aligned2 = read_igl_obj(
        input_mesh_path2)

    from fitting.measure import mesh2mesh, distance2color
    from sklearn.neighbors import NearestNeighbors

    from geometry import find_closest_point_with_barycentric
    nnpts_new, idxtri, bcpts = find_closest_point_with_barycentric(subdived_mesh2, subdived_mesh, subdived_mesh_f)

    #    neigh = NearestNeighbors(n_neighbors=1)
    #    neigh.fit(subdived_mesh)
    #    print 'neigh.fit'
    #    distances, indices = neigh.kneighbors(subdived_mesh2, return_distance=True)
    #    distance = signdistance(nnpts_new,subdived_mesh2,n_frame_aligned2)
    distance = mesh2mesh(subdived_mesh2, nnpts_new)
    print distance.max(), distance.min()
    vmax = distance.max()
    vmin = distance.min()
    color_3d = distance2color(dist=distance, vmin=0.1, vmax=0.3, cmap_name='jet')
    write_full_obj(subdived_mesh2, subdived_mesh_f2, n_frame_aligned2, n_f_frame_aligned2, t_frame_aligned2,
                   t_f_frame_aligned2,
                   color_3d,
                   output_mesh_path)
    # write_landmark_to_obj('E:/workspace/tencent_test/sceneobj/guanyu_landmark.obj', nnpts_new, size=10)


# 返回均值和方差
def get_mean_value(nlist):
    narray = np.array(nlist)
    sum1 = narray.sum()
    narray2 = narray * narray
    sum2 = narray2.sum()
    N = narray.size
    mean = sum1 / N
    var = sum2 / N - mean ** 2

    return mean, var, np.std(nlist, ddof=1)


def bacface_cull(V, F, V_N):
    new_F = []
    front_dir = np.array([0, 0, 1])
    for i in range(0, F.shape[0]):
        face_idx = F[i, :]
        #
        if face_idx.size < 3:
            continue
        n1 = V_N[face_idx[0]]
        n2 = V_N[face_idx[1]]
        n3 = V_N[face_idx[2]]
        if np.dot(n1, front_dir) < -0.5 and np.dot(n2, front_dir) < -0.5 and np.dot(n3, front_dir) < -0.5:
            continue
        new_F.append(face_idx)
    return V, np.array(new_F)


def merge_splitedmesh(input_obj_path, outputobjpath):
    subdived_mesh, subdived_mesh_f, t_frame_aligned, t_f_frame_aligned, n_frame_aligned, n_f_frame_aligned = read_igl_obj(
        input_obj_path)
    vtxid_map = {}  # 原来的顶点在修改后的顶点的位置
    vtx_hash_map = {}
    newvtx_idx_count = 0
    new_mesh_v = []
    for i in range(0, subdived_mesh.shape[0]):
        v = subdived_mesh[i]
        vtx_hash = str(v[0]) + "k" + str(v[1]) + "k" + str(v[2])
        if vtx_hash not in vtx_hash_map:
            vtx_hash_map[vtx_hash] = newvtx_idx_count  # 记录顶点的新位置
            vtxid_map[i] = newvtx_idx_count  # 记录原顶点在新顶点的位置
            new_mesh_v.append(v)
            newvtx_idx_count += 1
        else:
            vtxid_map[i] = vtx_hash_map[vtx_hash]

    new_mesh_f = []
    for i in range(0, subdived_mesh_f.shape[0]):
        face = subdived_mesh_f[i]
        new_face = [vtxid_map[face[0]], vtxid_map[face[1]], vtxid_map[face[2]]]
        new_mesh_f.append(new_face)
    new_mesh_v = np.array(new_mesh_v)
    new_mesh_f = np.array(new_mesh_f)
    get_vertex_normal(new_mesh_v, new_mesh_f)
    write_full_obj(new_mesh_v, new_mesh_f, np.array([]), np.array([]), np.array([]), np.array([]), np.array([]),
                   outputobjpath)


if __name__ == '__main__':
    pass
    # from fitting.util import  get_opt_transform_2d
    # landmark1 = np.array([[2,2,2],[3,3,3],[4,4,4],[1,1,1],[6,6,6],[7,7,7]])
    #
    # landmark2 = landmark1*4
    # scale,rotate,translate_2d =  get_opt_transform_2d(landmark1,landmark2[:,0:2])
    # print scale,rotate,translate_2d
    if 0:  # 调整网格方位
        project_dir = 'E:\workspace/vrn_data\wang/'
        out_dir = 'E:\workspace/vrn_data\wang/wang_crop/'
        project_dir = 'D:/program files/R3DS/Wrap 3.3/Models/Dataset_Subject1/Dataset_Subject1/images/select_mesh/obj/'
        out_dir = project_dir + '/convert/'
        b = FileFilt()
        b.FindFile(dirr=project_dir)
        import cv2

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
            if 0:
                bbox_2d = [300, 100, 900, 700]
                img = readImage(k)
                img = img[::-1, :, :]
                img = rescale_imge_with_bbox(img, bbox_2d, color=[0, 0, 0])
                img = img[::-1, :, :]
                cv2.imwrite(out_dir + str(obj_name).zfill(3) + '.png', img)
            v, f, t, t_f, n, n_f = read_igl_obj(k)
            v[:, 2] = -v[:, 2]
            v[:, 1] = -v[:, 1]
            v[:, 2] += 190
            v /= 90.0
            vc = readVertexColor(k)
            write_full_obj(mesh_v=v, mesh_f=f, mesh_n=n, mesh_n_f=n_f, mesh_tex=t, mesh_tex_f=t_f, vertexcolor=vc,
                           filepath=out_dir + '/' + obj_name + '.obj', generate_mtl=False, verbose=False,
                           img_name='default.png')
    if 0:  # 输出颜色
        v_1, f_1, t_1, t_f_1, n_1, n_f_1 = read_igl_obj(
            'D:\program files\R3DS\Wrap 3.3\Models\Dataset_Subject1\Dataset_Subject1\images\select_mesh\obj\convert/reconstruction_0246.obj')
        vc = readVertexColor(
            'D:\program files\R3DS\Wrap 3.3\Models\Dataset_Subject1\Dataset_Subject1\images\select_mesh\obj\convert/reconstruction_0246.obj')
        v_2, f_2, t_2, t_f_2, n_2, n_f_2 = read_igl_obj(
            'D:\program files\R3DS\Wrap 3.3\Models\Dataset_Subject1\Dataset_Subject1\images\select_mesh\obj\convert/reconstruction_0246_remap.obj')
        write_full_obj(mesh_v=v_1, mesh_f=f_1, mesh_n=n_1, mesh_n_f=n_f_1, mesh_tex=t_2, mesh_tex_f=t_f_2,
                       vertexcolor=vc,
                       filepath='D:\program files\R3DS\Wrap 3.3\Models\Dataset_Subject1\Dataset_Subject1\images\select_mesh\obj\convert/reconstruction_0246_remap_color.obj',
                       generate_mtl=False, verbose=False,
                       img_name='default.png')
    if 0:  # 加入眼球
        path = 'D:\program files\R3DS\Wrap 3.3\Models\Dataset_Subject1\Dataset_Subject1\images\select_mesh\obj\convert/align_flame/realign/'
        project_dir = path + 'realign_copy/'
        out_dir = path + '/realign_with_eye/'
        b = FileFilt()
        b.FindFile(dirr=project_dir)
        import cv2

        f_mask = [1284, 5212, 2173, 6085, 770, 4706, 894, 4827, 895, 4828, 901, 4834, 902, 892, 888, 4835, 4909, 916,
                  4848, 763, 4699, 765, 893, 977, 4911, 979, 4919, 984, 4916,
                  2339, 6250, 6249, 6245, 2335, 6248, 2338, 2337, 6144, 2232, 6143, 2231, 6723, 2814, 2441, 6351, 6354,
                  2444,
                  6353, 2442, 6352, 2360, 6270, 2230, 6142, 2229, 6247, 2336, 6141]
        v_eye, f_eye, t_eye, t_f_eye, n_eye, n_f_eye = read_igl_obj(path + 'eyeball_centered.obj')  # read eye ball
        right_eye_idx = 3930
        left_eye_idx = 3929
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

            v, f, t, t_f, n, n_f = read_igl_obj(k)
            tmp_f = f
            f = []
            for i in range(0, tmp_f.shape[0]):
                if i in f_mask:
                    continue
                f.append(tmp_f[i, :])
            f = np.array(f)
            right_eye_pos = v[right_eye_idx, :]
            left_eye_pos = v[left_eye_idx, :]
            right_eye_ball_v = v_eye + right_eye_pos
            left_eye_ball_v = v_eye + left_eye_pos
            v1, f1 = add_vertex_faces(v, f, left_eye_ball_v, f_eye)
            v2, f2 = add_vertex_faces(v1, f1, right_eye_ball_v, f_eye)
            # vc = readVertexColor(k)
            write_full_obj(mesh_v=v2, mesh_f=f2, mesh_n=n, mesh_n_f=n_f, mesh_tex=np.array([]), mesh_tex_f=np.array([]),
                           vertexcolor=np.array([]),
                           filepath=out_dir + '/' + obj_name + '.obj', generate_mtl=False, verbose=False,
                           img_name='default.png')
        pass

    if 0:

        path = 'D:\huayunhe/facewarehouse_new/'
        project_dir = path + 'FaceWarehouse_Tester101-150'  # 'FaceWarehouse_Tester51-100' #'FaceWarehouse_Tester1-50/'
        out_dir = path + '/FaceWarehouse_neutral_img/'
        out_obj_dir = path + '/FaceWarehouse_neutral_img_obj/'

        dirr = project_dir
        for s in os.listdir(dirr):
            newDir = os.path.join(dirr, s)
            if os.path.isfile(newDir):
                pass
            elif os.path.isdir(newDir):
                for s2 in os.listdir(newDir):
                    if s2 == 'TrainingPose':
                        newDir2 = newDir + '/' + s2 + '/'
                        img_path = newDir2 + 'pose_0.png'
                        obj_path = newDir2 + 'pose_0.obj'
                        output_img_path = out_dir + s + '_' + 'pose_0.png'
                        output_obj_path = out_obj_dir + s + '_' + 'pose_0.obj'
                        moveFileto(img_path, output_img_path)
                        moveFileto(obj_path, output_obj_path)
                    pass

        b = FileFilt()
        b.FindFile(dirr=project_dir)
        for k in b.fileList:
            if k == '':
                continue
            filename_split = k.split("/")[-1].split(".")

        pass

    if 1:
        mesh_error_compare(
            input_mesh_path1='D:\huayunhe/facewarehouse_new\FaceWarehouse_neutral_img_obj//Tester_1_pose_0.obj',
            input_mesh_path2='D:\huayunhe/facewarehouse_new\FaceWarehouse_neutral_img_obj/align/align000.obj',
            output_mesh_path='D:\huayunhe/facewarehouse_new\FaceWarehouse_neutral_img_obj/output/out1.obj')

    if 0:
        # merge_splitedmesh(input_obj_path='E:/workspace/tencent_test/sceneobj/capsule2.obj', outputobjpath='E:/workspace/tencent_test/sceneobj/capsule_merge.obj')
        # merge_splitedmesh(input_obj_path='E:/workspace/tencent_test/sceneobj/guanyu.obj', outputobjpath='E:/workspace/tencent_test/sceneobj/guanyu_merge.obj')
        merge_splitedmesh(input_obj_path='E:/workspace/tencent_test/sceneobj/guanyucollider.obj',
                          outputobjpath='E:/workspace/tencent_test/sceneobj/guanyucollider_merge.obj')
    if 0:
        mesh_error_compare2(input_mesh_path2='E:/workspace/tencent_test/sceneobj/guanyu_merge.obj',
                            input_mesh_path1='E:/workspace/tencent_test/sceneobj/guanyucollider_merge.obj',
                            output_mesh_path='E:/workspace/tencent_test/sceneobj/guanyu_compare.obj')
    if 0:
        mesh_error_compare2(input_mesh_path2='E:/workspace/tencent_test/sceneobj/guanyu_merge_fillhole_union_clean.obj',
                            input_mesh_path1='E:/workspace/tencent_test/sceneobj/guanyucollider_merge_union2.obj',
                            output_mesh_path='E:/workspace/tencent_test/sceneobj/guanyu_compare.obj')
    if 0:
        merge_splitedmesh(input_obj_path='E:/workspace/tencent_test/sceneobj/MPLevel03.obj',
                          outputobjpath='E:/workspace/tencent_test/sceneobj/MPLevel03_merge.obj')
