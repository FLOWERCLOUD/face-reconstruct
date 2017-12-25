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
    with open( filepath) as f:
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
        for f in mesh_f+1: # Faces are 1-based, not 0-based in obj files
            fp.write( 'f %d %d %d\n' %  (f[0], f[1], f[2]) )
    if verbose:
        print 'mesh saved to: ', filepath 

# -----------------------------------------------------------------------------

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
import cv2
def write_image_and_featurepoint(image,feature_point,file_path):
    for (x, y) in feature_point:
        cv2.circle(image, (int(x), int(y)), 1, (0, 0, 255), -1)
    cv2.imwrite(file_path, image)
def readImage(path):
    image = cv2.imread(path,cv2.IMREAD_COLOR)
    return image
def sym_point( front,up,normal,mean,point):
    m = np.vstack([front,up,normal]) #投射矩阵
    m2 = np.hstack([front,up,normal]) #基矩阵
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
    result = v1.dot(v3) -v2.dot(v4);
import os
Const_Image_Format = [".jpg",".jpeg",".bmp",".png"]
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
''' use example
if __name__ == "__main__":
        b = FileFilt()
        b.FindFile(dirr = "D:\Python27\user\dapei-imgs")
        print(b.counter)
        for k in b.fileList:
            print k
'''

