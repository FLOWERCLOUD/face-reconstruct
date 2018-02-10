# -- coding: utf-8 --
'''
Util funcitons - general
Tianye Li <tianye.li@tuebingen.mpg.de>
'''

import os
import numpy as np
import cPickle as pickle
import scipy.io as scio
import sys
sys.path.insert(0, "D:/mprojects/libiglfull/libigl/python/build/x64/Release")
import pyigl as igl

# -----------------------------------------------------------------------------

def load_binary_pickle( filepath ):
    # type: (object) -> object
    with open( filepath,'rb') as f:
        data = pickle.load( f )
    return data
def load_pickle( filepath ):
    # type: (object) -> object
    with open( filepath,) as f:
        data = pickle.load( f )
    return data

# -----------------------------------------------------------------------------

def save_binary_pickle( data, filepath ):
    with open( filepath, 'wb' ) as f:
        pickle.dump( data, f )

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

        if str[0] ==  '#':
            continue
        elif str[0] == 'v':
            tmp_v = [get_num(s, 'float') for s in str[1:]]
            vertices.append( tmp_v )

        elif str[0] == 'f':
            tmp_f = [get_num(s, 'int')-1 for s in str[1:]]
            faces.append( tmp_f )

    f.close()
    return ( np.asarray(vertices), np.asarray(faces) )

# -----------------------------------------------------------------------------

def write_simple_obj( mesh_v, mesh_f, filepath, verbose=False ):
    with open( filepath, 'w') as fp:
        for v in mesh_v:
            fp.write( 'v %f %f %f\n' % ( v[0], v[1], v[2]) )
        for f in mesh_f: # Faces are 1-based, not 0-based in obj files
            fp.write( 'f %d %d %d\n' %  (f[0]+1, f[1]+1, f[2]+1) )
    if verbose:
        print 'mesh saved to: ', filepath

def write_full_obj(mesh_v, mesh_f, mesh_n,mesh_n_f,mesh_tex,mesh_tex_f,vertexcolor,filepath, generate_mtl=False,verbose=False,img_name = 'default.png'):
    with open(filepath, 'w') as fp:

        if generate_mtl:
            mtlfile = filepath+'.mtl'
            with open(mtlfile, 'w') as fmtl:
                fmtl.write('newmtl material_0\n')
                fmtl.write('Ka 0.200000 0.200000 0.200000\n')
                fmtl.write('Kd 0.000000 0.000000 0.000000\n')
                fmtl.write('Ks 1.000000 1.000000 1.000000\n')
                fmtl.write('Tr 0.000000\n')
                fmtl.write('illum 2\n')
                fmtl.write('Ns 0.000000\n')
                fmtl.write('map_Kd '+img_name+ '\n')
            mtlname = mtlfile.split('/')[-1]
            fp.write('mtllib '+'./'+ mtlname+'\n')
        count_v = 0
        for v in mesh_v:
            if vertexcolor.size > 0:
                fp.write('v %f %f %f %f %f %f\n' % (v[0], v[1], v[2],
                       vertexcolor[count_v,0],vertexcolor[count_v,1],vertexcolor[count_v,2]))
            else:
                fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
            count_v+=1
        for v in mesh_tex:
            fp.write('vt %f %f\n' % (v[0], v[1]))
        for v in mesh_n:
            fp.write('vn %f %f %f\n' % (v[0], v[1], v[2]))
        count_f = 0
        for f in mesh_f + 1:  # Faces are 1-based, not 0-based in obj files
            fp.write('f %d/' % (f[0]))
            if mesh_tex_f.size > 0:
                fp.write('%d' % (mesh_tex_f[count_f,0]+ 1) )
            else:
                pass
            fp.write('/')
            if mesh_n_f.size > 0:
                fp.write('%d' % (mesh_n_f[count_f,0]+ 1))
            else:
                pass
            fp.write(' ')
            fp.write('%d/' % (f[1]))
            if mesh_tex_f.size > 0:
                fp.write('%d' % (mesh_tex_f[count_f,1]+ 1))
            else:
                pass
            fp.write('/')
            if mesh_n_f.size > 0:
                fp.write('%d' % (mesh_n_f[count_f,1]+ 1))
            else:
                pass
            fp.write(' ')
            fp.write('%d/' % (f[2]))
            if mesh_tex_f.size > 0:
                fp.write('%d' % (mesh_tex_f[count_f,2]+ 1))
            else:
                pass
            fp.write('/')
            if mesh_n_f.size > 0:
                fp.write('%d' % (mesh_n_f[count_f,2]+ 1))
            else:
                pass
            fp.write('\n')

            count_f+=1
        if verbose:
            print 'mesh saved to: ', filepath


        # -----------------------------------------------------------------------------
def readVertexColor(path):
    result =[]
    with open(path, 'r') as f:
        data = f.readlines()  # txt中所有字符串读入data
        for line in data:
            odom = line.split(' ')  # 将单个数据分隔开存好
            if len(odom) >0 and odom[0] =='v':
               if len(odom) >6:
                    c1 = float(odom[4])  #
                    c2 = float(odom[5])  #
                    c3 = float(odom[6])  #
                    result.append([c1,c2,c3])
    return  np.array(result)


def safe_mkdir( file_dir ):
    if not os.path.exists( file_dir ):
        os.mkdir( file_dir )


def pickle_save(contact,filepath):  #使用pickle模块将数据对象保存到文件
    f = open(filepath, 'w')
    pickle.dump(contact, f)
    f.close()


def pickle_load(filepath):  # 使用pickle从文件中重构python对象
    f = open(filepath, 'r')
    contact = pickle.load(f)
    f.close()
    return contact

def mat_save(contact,filepath):  #存储为mat文件
    scio.savemat(filepath, contact)
def convertObj2Mat(objpath,matpath,istarget):
    v = igl.eigen.MatrixXd()
    f = igl.eigen.MatrixXi()
    n = igl.eigen.MatrixXd()
    n_f = igl.eigen.MatrixXi()
    t = igl.eigen.MatrixXd()
    t_f = igl.eigen.MatrixXi()
    igl.readOBJ(objpath, v,t,n,f,t_f,n_f)
    v = np.array(v)
    f = np.array(f)
    n = np.array(n)
    if istarget:
        Target = {'vertices': v, 'faces': f + 1, 'normals': n}
        Target = {'Target': Target}
        mat_save(Target, matpath)
    else :
        Source = {'vertices': v, 'faces': f + 1, 'normals': n}
        Source = {'Source': Source}
        mat_save(Source, matpath)

def convertMat2obj(matpath,objpath,str):
    contact = mat_load(matpath)
    target = contact[str]
    vertices = target['vertices'][0, 0]
    faces = target['faces'][0, 0]
    normals = target['normals'][0, 0]
    faces =faces -1
    igl.writeOBJ(objpath, igl.eigen.MatrixXd(vertices.astype('float64')),
                    igl.eigen.MatrixXi(faces.astype('intc')),igl.eigen.MatrixXd(normals.astype('float64')),
                 igl.eigen.MatrixXi(faces.astype('intc')),igl.eigen.MatrixXd(),igl.eigen.MatrixXi())

def mat_load(filepath):  # 读取mat文件
    contact = scio.loadmat(filepath,struct_as_record=True)
    return contact

def npArrayToIglMatrixi(input):
    e2 = igl.eigen.MatrixXi()
    e2.resize(input.shape[0], input.shape[1])
    for i in range(0, input.shape[0]):
        for j in range(0, input.shape[1]):
            e2[i, j] = input[i][j]
    return e2
def IglMatrixTonpArray(input):
    #e2 = np.arange(input.shape[0]*input.shape[1]).reshape(input.shape[0], input.shape[1])
    e2 = np.array(input)
    '''
    for i in range(0, input.shape[0]):
        for j in range(0, input.shape[1]):
            e2[i][j] = input[i, j]
    '''
    return e2
#https://www.jianshu.com/p/4528aaa6dc48
def k_main_dir(Mesh,K):
    mean = np.mean(Mesh)
    res = Mesh -mean

from  sklearn.decomposition import PCA
def k_main_dir_sklearn(Mesh,K=3):
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
        print 'eigenvalue',(eigenvalue)
        print 'eigenvector',eigenvector
        result.append(eigenvector)
    result = np.array(result)
    return np.mean(Mesh, axis=0),result,eigenvalues

def save_landmark(file_path,landmark):
    with open(file_path, 'w') as file:
        if( landmark.shape[1] ==2):
            for i in range( 0,landmark.shape[0]):
                write_str = '%f %f\n' % (landmark[i,0], landmark[i,1])
                file.write(write_str)

        elif landmark.shape[1] ==3:
            for i in range( 0,landmark.shape[0]):
                write_str = '%f %f %f\n' % (landmark[i,0], landmark[i,1],landmark[i,2])
                file.write(write_str)
        elif landmark.shape[1] ==1:
            for i in range( 0,landmark.shape[0]):
                write_str = '%f\n' % (landmark[i,0])
                file.write(write_str)
def save_int(file_path,landmark):
    with open(file_path, 'w') as file:
        if( landmark.shape[1] ==2):
            for i in range( 0,landmark.shape[0]):
                write_str = '%d %d\n' % (landmark[i,0], landmark[i,1])
                file.write(write_str)

        elif landmark.shape[1] ==3:
            for i in range( 0,landmark.shape[0]):
                write_str = '%d %d %d\n' % (landmark[i,0], landmark[i,1],landmark[i,2])
                file.write(write_str)
        elif landmark.shape[1] ==1:
            for i in range( 0,landmark.shape[0]):
                write_str = '%d\n' % (landmark[i,0])
                file.write(write_str)
def read_int(file_path):
    result =[]
    with open(file_path, 'r') as f:
        data = f.readlines()  # txt中所有字符串读入data
        for line in data:
            odom = line.split(' ')  # 将单个数据分隔开存好
            numbers_float = map(int, odom)  # 转化为整数
            result.append(numbers_float)
    return  np.array(result)

def read_landmark(file_path):
    result =[]
    with open(file_path, 'r') as f:
        data = f.readlines()  # txt中所有字符串读入data
        for line in data:
            odom = line.split(',')  # 将单个数据分隔开存好
            numbers_float = map(float, odom)  # 转化为浮点数
            result.append(numbers_float)
    return  np.array(result)
#或者 a = numpy.loadtxt('odom.txt')
def add_vertex_faces(pre_v,pre_f,add_v,add_f):
    if pre_v.size ==0:
        new_v = add_v
        new_f = add_f
    else:
        new_v = np.vstack((pre_v, add_v))
        pre_num_vertices = pre_v.shape[0]
        add_f = add_f+pre_num_vertices
        new_f =  np.vstack((pre_f,add_f))
    return new_v,new_f

def write_landmark_to_obj(file_path,landmark,size =1000):
    sphere_v = igl.eigen.MatrixXd()
    sphere_f = igl.eigen.MatrixXi()
    igl.readOBJ('sphere.obj', sphere_v,
                sphere_f)
    sphere_v = np.array(sphere_v)
    sphere_f = np.array(sphere_f)
    lmk_num = landmark.shape[0]
    sphere_v_move =np.array([])
    all_v =np.array([])
    all_f =np.array([])
    for i in range(0, lmk_num):
        sphere_v_move = size*sphere_v + landmark[i, :]
        all_v,all_f = add_vertex_faces(all_v,all_f,sphere_v_move,sphere_f)
    igl.writeOBJ(file_path, igl.eigen.MatrixXd(all_v),
                 igl.eigen.MatrixXi(all_f.astype('intc')))

def numpy_load(file_path):
    a = np.loadtxt(file_path)
    return  a
def triangle_intersect(origin, dir,v1,v2,v3):
    origin  = np.array(origin)
    dir = np.array(dir)
    v1 = np.array(v1)
    v2 = np.array(v2)
    v3 = np.array(v3)
    isintersect = False
    t = 0.0
    dir = dir/np.linalg.norm(dir)
    EPSILON = 10e-6
    result_point = []
    e1 = v2 - v1
    e2 = v3 - v1
    p = np.cross(dir ,e2)
    tmp = p.dot(e1)
    if tmp > - EPSILON and tmp < EPSILON :
        return isintersect,result_point,t
    tmp  = 1.0/tmp
    s = origin - v1
    u = tmp *(s.dot(p))
    if (u < 0.0 or u >1.0):
        return isintersect,result_point,t
    q = np.cross(s,e1)
    v = tmp *(dir.dot(q))
    if v <0.0 or v> 1.0:
        return isintersect,result_point,t
    uv = u +v
    if uv > 1.0:
        return isintersect, result_point, t
    t = tmp *(e2.dot(q))
    if t <0:
        isintersect = False
        return isintersect,result_point,t
    isintersect = True
    result_point = origin + t * dir
    return  isintersect,result_point,t
def cast2d_to3d(point2d,v_mesh,f_mesh):
    result =[]
    min_t = 2000
    for i in range(0,f_mesh.shape[0]):
        isintersect,intersect_point,t = triangle_intersect([point2d[0],point2d[1],192],[0,0,-1],
                                                           v_mesh[f_mesh[i,0],:],v_mesh[f_mesh[i,1],:],v_mesh[f_mesh[i,2],:])
        if isintersect == True:
            if t>0:
                if t <min_t:
                    min_t = t
                    result = intersect_point
    if len(result)>0:
        issucess = True
    else:
        issucess = False

    return issucess,result
#use trimesh

def cast2d_to3d_trimesh(mesh_path,point2d):
    import trimesh
    mesh = trimesh.load(mesh_path)
    point_num = point2d.shape[0]
    ray_origins = np.zeros((point_num, 3), dtype=np.float64)
    ray_directions = np.zeros((point_num, 3), dtype=np.float64)
    for i in range(0,point_num):
        ray_origins[i]= np.array([point2d[i,0],point2d[i,1],192])
        ray_directions[i] = np.array([0, 0, -1])
    index_triangles, index_ray, result_locations = mesh.ray.intersects_id(ray_origins=ray_origins,
                                                                          ray_directions=ray_directions,

                                                                          multiple_hits=False, return_locations=True)
    return index_triangles,index_ray,result_locations




def scaleToOriCoodi_topleft(source ,BB , scaled_size):
    if BB[2]!=BB[3]:
        print 'BB has error'
    ori_size = BB[2]
    target = source/float(scaled_size)*ori_size
    target = target + np.array([BB[0],BB[2]])
    return  target
def scaleToOriCoodi_bottomleft(source ,BB , scaled_size):
    if BB[2]!=BB[3]:
        print 'BB has error'
    ori_size = BB[2]
    target = source/float(scaled_size)*ori_size
    target[:,0:2] = target[:,0:2]+ np.array([BB[0],BB[1]])
    return  target

def write_image_and_featurepoint(image,feature_point,file_path):
    import cv2
    image = image.copy()
    count  = 0
    for (x, y) in feature_point:
        if count <8:
            cv2.circle(image, (int(x), int(y)), 1, (0, 0, 255), -1)
        elif  count >8 and count <17:
            cv2.circle(image, (int(x), int(y)), 1, (0, 255, 0), -1)
        elif  count in [27,28,29,30,33,51,62,66,57,8]:
            cv2.circle(image, (int(x), int(y)), 1, (255, 255, 0), -1)
        elif  count in [17,18,19,20,21,36,37,38,39,40,41]:
            cv2.circle(image, (int(x), int(y)), 1, (0, 255, 255), -1)
        elif  count in [22 ,23,24,25,26,42,43,44,45,46,47]:
            cv2.circle(image, (int(x), int(y)), 1, (255, 0, 255), -1)
        elif  count in [48,49,50,60,61,67,58,59]:
            cv2.circle(image, (int(x), int(y)), 1, (255, 0,0), -1)
        elif  count in [52,53,54,55,56,63,64,65]:
            cv2.circle(image, (int(x), int(y)), 1, (0, 255, 255), -1)
        elif  count in [31,32]:
            cv2.circle(image, (int(x), int(y)), 1, (0, 255, 0), -1)
        elif  count in [34,35]:
            cv2.circle(image, (int(x), int(y)), 1, (128, 0,128), -1)
        count+=1
    cv2.imwrite(file_path, image)


def detect_68_landmark_dlib(image): #左上角为原点
    import dlib
    height = image.shape[0]
    width = image.shape[1]
    predictor_path = 'D:/mprojects/FaceProject/FaceProject/src/shape_predictor_68_face_landmarks.dat'
    predictor_path = 'shape_predictor_68_face_landmarks.dat'
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    dets = detector(image, 1)
    featurePointxy = np.zeros([68, 2])
    if len(dets) >=0:
        shape = predictor(image, dets[0])
    for i in xrange(68):
        featurePointxy[i, 0] = shape.part(i).x
        featurePointxy[i, 1] = shape.part(i).y
    return    featurePointxy

def readImage(path):
    #cv2.IMREAD_UNCHANGED 四通道
    image = cv2.imread(path,cv2.IMREAD_COLOR)
    return image
def sym_point( front,up,normal,mean,point):
    m = np.vstack([front,up,normal]) #投射矩阵
    front= front.reshape( front.size,1)
    up = up.reshape(up.size, 1)
    normal = normal.reshape(normal.size, 1)
    m2 = np.concatenate((front,up,normal), axis=1)
    project_coff = np.dot(m,(point-mean))  #使用 .dot才是矩阵乘法，否则是对应相乘
    project_coff[2] = -project_coff[2] #front,up 平面对称
    new_point = m2.dot(project_coff)+mean
    return new_point

def corr_point( front,up,normal,mean,point):
    m = np.vstack([front,up,normal]) #投射矩阵
    front= front.reshape( front.size,1)
    up = up.reshape(up.size, 1)
    normal = normal.reshape(normal.size, 1)
    m2 = np.concatenate((front,up,normal), axis=1)
#    m2 = np.hstack([front.T,up.T,normal.T]) #基矩阵
    project_coff = np.dot(m,(point-mean))  #使用 .dot才是矩阵乘法，否则是对应相乘
    project_coff[2] = 0 #front,up 平面对称
    new_point = m2.dot(project_coff)+mean
    return new_point

def corr_landmark_tofit_data(front,up,normal,mean,mesh,landmark):
    if( len(landmark.shape) == 1):
       landmark = landmark.reshape(1,landmark.size)
    m = np.vstack([front,up,normal]) #投射矩阵
    centered_mesh = mesh-mean
    project_coff = np.dot(m,(centered_mesh.T))  #使用 .dot才是矩阵乘法，否则是对应相乘
    project_coff = project_coff.T

    up_min  = min(project_coff[:,1])
    up_max  = max(project_coff[:,1])
    grid_num =200
    depth_buf = np.zeros((grid_num,1),np.float64)
    step = (up_max-up_min)/grid_num
    for i in range(0,project_coff.shape[0]):
        #去掉偏离中心面的点
        if abs(project_coff[i, 2]) >3:
            continue
        curr_depth = project_coff[i,0]
        grid_id = int((project_coff[i,1]-up_min)/step)
        if grid_id <0:
            grid_id =0
        if grid_id > grid_num-1:
            grid_id = grid_num-1
        if depth_buf[grid_id] < curr_depth:
            depth_buf[grid_id] =  curr_depth
#    print  depth_buf

    landmark = landmark- mean
    project_landmark = np.dot(m,(landmark.T))
    project_landmark = project_landmark.T
    for i in range(0,project_landmark.shape[0]):
        grid_id = int((project_landmark[i,1]-up_min)/step)
        if grid_id <0:
            grid_id =0
        if grid_id > grid_num-1:
            grid_id = grid_num-1
        project_landmark[i,0] = depth_buf[grid_id]
    front= front.reshape( front.size,1)
    up = up.reshape(up.size, 1)
    normal = normal.reshape(normal.size, 1)
    m2 = np.concatenate((front,up,normal), axis=1)
    project_landmark = m2.dot(project_landmark.T)
    project_landmark = project_landmark.T +mean

    buf_vertex=[]
    for i in range(0,200):
        buf_vertex.append([depth_buf[i],up_min+step*i,0])
    buf_vertex = np.array(buf_vertex)
    buf_vertex = m2.dot(buf_vertex.T)
    buf_vertex = buf_vertex.T +mean

    return project_landmark,buf_vertex



def sym_plane( front,up,normal,mean,scale):
    vertex = np.zeros((9,3))
    vertex[0, :] = mean+front*scale+up*scale
    vertex[1, :] = mean+front * scale
    vertex[2, :] = mean+[0,0,0]
    vertex[3, :] = mean+up*scale
    vertex[4, :] = mean+-front * scale
    vertex[5, :] = mean+-front * scale+up*scale
    vertex[6, :] = mean+front * scale - up * scale
    vertex[7, :] = mean+- up * scale
    vertex[8, :] = mean+-front * scale- up * scale
    vertex = np.array(vertex)
    f = [[0,1,2],[0,2,3],[3,2,4],[3,4,5],[1,6,7],[1,7,2],[2,7,8],[2,8,4]]
    f = np.array(f)
    return vertex,f

def cast_coeff(front ,point):
    coeff = front.dot(point)
    return coeff



def  determinant( v1,v2,v3,v4 ):
    result = v1.dot(v3) -v2.dot(v4)

def convert_para_tex(para_objmesh, paraTexMat):

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

    for i_pc in range(0,1):
        '''
        coff = np.zeros((199,1))
        coff[i_pc] = 5e3
        cur_tex = texMU+np.dot(texPC,coff)
        cur_tex = cur_tex.reshape(cur_tex.size/3,3)
        output_path = './texpc/' + 'pc_' + str(i_pc) + '_coff_' + str(float(coff[i_pc])) + '.obj'
        write_full_obj(v,f,n,n_f,t,t_f,cur_tex/255.0,output_path)
        '''
        coff = np.zeros((199,1))
        coff[i_pc] =4.1031792e3*10
        print coff.shape
        coff.reshape(coff.size,1)
        print coff.shape
        cur_tex = texMU+np.dot(texPC,coff)
        for i in range(0,cur_tex.size):
            if cur_tex[i]<0.0:
                cur_tex[i] =0.0
            if cur_tex[i] > 255.0:
                cur_tex[i] = 255.0
            cur_tex[i]/=255.0

        cur_tex = cur_tex.reshape(cur_tex.size/3,3)

        output_path = './texpc/' + 'pc_' + str(i_pc) + '_coff_' + str(float(coff[i_pc])) + '.obj'
        write_full_obj(v,f,n,n_f,t,t_f,cur_tex,output_path)

    '''
        igl.writeOBJ(output_path,
                    igl.eigen.MatrixXd(v.astype('float64')),igl.eigen.MatrixXi(f.astype('intc')),
                    igl.eigen.MatrixXd(n.astype('float64')),igl.eigen.MatrixXi(n_f.astype('intc')),
                    igl.eigen.MatrixXd(t.astype('float64')),t_f.astype('intc'))

    '''



Const_Image_Format = [".jpg",".jpeg",".bmp",".png",".obj",".ppm",".data"]
class FileFilt:

    fileList = [""]
    counter = 0
    def __init__(self):
        pass
    def FindFile(self,dirr,filtrate = 1):
        global Const_Image_Format
        for s in os.listdir(dirr):
            newDir = os.path.join(dirr,s)
            if os.path.isfile(newDir):
                if filtrate:
                        if newDir and(os.path.splitext(newDir)[1] in Const_Image_Format):
                            self.fileList.append(newDir)
                            self.counter+=1
                else:
                    self.fileList.append(newDir)
                    self.counter+=1
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
    result =[]
    with open(maya_path, 'r') as f:
        data = f.readlines()  # txt中所有字符串读入data
        for line in data:
            odom = line.split(' ')  # 将单个数据分隔开存好
            for part in odom:
                left_quo = part.find('[')
                right_quo = part.find(']')
                if left_quo >-1 and right_quo >-1:
                    extract = part[left_quo+1:right_quo]
                    maohao_pos = extract.find(':')
                    if maohao_pos >-1:
                        left_pos = extract[0:maohao_pos]
                        right_pos = extract[maohao_pos+1:len(extract)]
                        left_pos = int(left_pos)
                        right_pos = int(right_pos)
                        for pos in range(left_pos,right_pos+1):
                            result.append(pos)
                    else:
                        pos = int(extract)
                        result.append(pos)
    return  np.array(result)

def cac_normal(v_np,f_np):
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
    t = np.array(t)
    n = np.array(n)
    f = np.array(f)
    t_f = np.array(t_f)
    n_f = np.array(n_f)
    return v,f,t,t_f,n,n_f
