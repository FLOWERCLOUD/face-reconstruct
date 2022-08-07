# -- coding: utf-8 --
import scipy.io as sio
import numpy as np
import sys
from configs.config import igl_python_path
sys.path.insert(0, igl_python_path)
import pyigl as igl
from sklearn.decomposition import PCA
from scipy import sparse
import scipy.sparse.linalg as LSA
import numpy.linalg as LA
from scipy import optimize
def npArrayToIglMatrixi(input):
    e2 =igl.eigen.MatrixXi(input.astype('intc'))
    return e2
h_4 = 0.01 #0.08
h_5 = 0.008#0.00#h_5 = 0.08

h_6 = 0.00# h_6 = 0.06
h_7 = 0.00# h_7 = 0.06
h_8 = 0.00# h_8 = 0.06

class FaceModel(object):
    def __init__(self,  ShapeMU, Tl=None):
        self.base_mesh = ShapeMU
        self.shapeMU = ShapeMU
        self.n_vertices = ShapeMU.shape[0]
        self.faceshape = self.shapeMU
        self.tl = Tl
        pass
    def generate_f_list(self,
                        expression_file_path = 'D:\mproject/face-reconstruct\output/flame_blendshape/test_transfer/expression.npz'):
        """
        根据人脸
        """
        LLL = np.load(expression_file_path)
        self.express_GF_list = LLL['express_GF_list']
        express_GTGF_list = LLL['express_GTGF_list']
        self.n_expresses = len(self.express_GF_list)
        self.express_f_list = []
        for i in xrange(self.n_expresses):
            print 'express',i,len(express_GTGF_list)
            self.express_f_list.append(LSA.splu(express_GTGF_list[i]))
#        self.Express_displacement = np.zeros([self.n_components,self.n_expresses, self.n_vertices, 3 ])
        self.blendshape = np.zeros([self.n_expresses, self.n_vertices, 3 ])
        return
    def compute_express0 (self, alpha, z0):
        """
        input:
        alpha: 自然表情的形状参数 1 * n_components
        z0: 自然表情的Laplacian谱 1 * n_L_bases
        output:generate_blendshape_from_b0
        faceshape: 自然表情下的脸部形状
        """
        faceshape = self.base_mesh +  + (z0 * (self.L_EV).reshape(1,self.n_L_bases)).dot(self.L.reshape(self.n_L_bases, self.n_vertices * 3)).reshape(self.n_vertices, 3)
        return faceshape
    def generate_blendshape_from_b0 (self, b0):
        """
        input:
        b0: 自然表情下的脸部形状
        output:
        self.blendshape： 由自然表情生成的各种表情形状
        """
        from time import time
        timer_start = time()
        self.b0 = b0
        for i in xrange(self.n_expresses):
            self.blendshape[i, :, :] = self.express_f_list[i].solve(self.express_GF_list[i].dot(b0.reshape(self.n_vertices * 3))).reshape(self.n_vertices, 3) - b0
        timer_end = time()
        print "generate_blendshape_from_b0 in %f sec\n" % (timer_end - timer_start)
        return
    def compute_shape_from_blendshape(self, beta):
        faceshape = self.b0 + (beta.dot(self.blendshape.reshape(self.n_expresses, self.n_vertices * 3))).reshape(self.n_vertices, 3)
        return faceshape

    def compute_Laplacian(self):
        """
        求出平均形状的Laplacian算子
        """
        L = igl.eigen.SparseMatrixd()

        # edited by huayun
        # e2=igl.eigen.MatrixXi()
        # e2.resize(self.tl.shape[0],self.tl.shape[1])
        # for i in range(0,self.tl.shape[0]):
        #     for j in range(0, self.tl.shape[1]):
        #         e2[i,j] = self.tl[i][j]
        e2 = npArrayToIglMatrixi(self.tl)
        igl.cotmatrix(igl.eigen.MatrixXd(self.base_mesh), e2, L)
        #        igl.cotmatrix(igl.eigen.MatrixXd(self.shapeMU), igl.eigen.MatrixXi(self.tl), L)
        """
        求出Laplacian算子的eigenVector
        """
        L = L.toCOO()
        HH = np.array(L)
        n_rows, n_cols = HH.shape
        x_list = np.array(HH[:, 0], dtype='int32')
        y_list = np.array(HH[:, 1], dtype='int32')
        z_list = np.array(HH[:, 2], dtype='float64')
        LaplacianMatrix = sparse.csr_matrix((z_list, (x_list, y_list)), shape=(self.n_vertices, self.n_vertices))
        u, s, v = LSA.svds(LaplacianMatrix.T.dot(LaplacianMatrix), 50)
        Laplacian_bases = np.zeros([50, self.n_vertices, 3])
        self.n_L_bases = 50
        for i in xrange(self.n_L_bases):
            HH = v[i, :].reshape([1, self.n_vertices]).dot(self.shapeMU)
            Laplacian_bases[i, :, :] = u[:, i].reshape([self.n_vertices, 1]).dot(HH)
        self.L = Laplacian_bases
        self.L_EV = np.sqrt(s)

def rodrigues(x):
    import cv2
    return cv2.Rodrigues(x)[0]

def get_shape_3d(faceshape,mesh_face,lmk_face_idx, lmk_b_coords, lmk_facevtx_idx):
    from fitting.util import mesh_points_by_barycentric_coordinates
    v_selected_3d = mesh_points_by_barycentric_coordinates(faceshape, mesh_face, lmk_face_idx, lmk_b_coords)
    use_lunkuo = 1
    source_face_lmkvtx = faceshape[lmk_facevtx_idx[0:17]]
    if use_lunkuo:
        frame_landmark_idx = range(0, 17) + range(17, 60) + range(61, 64) + range(65, 68)
    else:
        frame_landmark_idx = range(17, 60) + range(61, 64) + range(65, 68)
    if use_lunkuo:
        v_selected_merge = np.vstack([source_face_lmkvtx, v_selected_3d])
    else:
        v_selected_merge = v_selected_3d
    return v_selected_merge
def optexpressFromRST_t(face_model, mesh_face,R, s, t, featurePointxy, lmk_face_idx, lmk_b_coords, lmk_facevtx_idx, beta_pre, boundShape,cur_frame_num = 0):
    import numpy.linalg as LA
    """
    Initialize camera translation and body orientation
    :param face: face model
    :param j2d: 2 * 68 array of face feature points
    :param  beta_pre 上一帧的表情系数
    :param cur_frame_num 如果这个为0 ，即不用考虑 beta_pre
    :param 网格上68个特征点的idx
    :returns: 表情系数,
    """
    face = face_model
    proj = np.array([[1, 0, 0], [0, 1, 0]])
    cameraProj = s * (proj.dot(R))
    #cameraTrans = s * t
    cameraTrans = t # 我的t是在scale 外面的，因而不用乘s
    #n_fps = fp_index.shape[0]
    n_fps = 66

    A = np.zeros([2 * n_fps, face.n_expresses])
    b = np.zeros(2 * n_fps)
    c = beta_pre.reshape(face.n_expresses)
    # selected_b0 = face.b0[fp_index, :]
    # selected_blendshape = face.blendshape[:, fp_index, :]
    selected_b0 = get_shape_3d(face.b0,mesh_face,lmk_face_idx, lmk_b_coords, lmk_facevtx_idx)
    selected_blendshape = np.zeros((face.blendshape.shape[0],n_fps,face.blendshape.shape[2]))
    for i in range(0,face.blendshape.shape[0]):
        selected_blendshape[i,:,:] = get_shape_3d( face.blendshape[i,:,:],mesh_face,lmk_face_idx, lmk_b_coords, lmk_facevtx_idx)
    frame_landmark_idx = range(0, 17) + range(17, 60) + range(61, 64) + range(65, 68)
    featurePointxy = featurePointxy[frame_landmark_idx,:]
    for i in xrange(n_fps):
        h = featurePointxy.T[:, i] - (cameraProj.dot(selected_b0[i, :].reshape([3, 1]))).reshape(2) - cameraTrans.reshape(2)
        b[2 * i + 0] = h[0]
        b[2 * i + 1] = h[1]
        for j in xrange(face.n_expresses):
            HH = cameraProj.dot(selected_blendshape[j, i, :].reshape([3, 1]))

            A[2 * i + 0, j] = HH[0, 0]
            A[2 * i + 1, j] = HH[1, 0]
    A = boundShape * A
    b = boundShape * b
    global h_4, h_5
    B = h_4 * np.eye(face.n_expresses)
    C = h_5 * np.eye(face.n_expresses)
    AA = A.T.dot(A) + B.T.dot(B) + C.T.dot(C)
    bb = A.T.dot(b) + h_5 * (C.T.dot(c))
    beta = LA.lstsq(AA, bb)[0]
    return beta.reshape([1, face.n_expresses])
def featureEnergyGivenshape(faceshape,mesh_face, pi, lamb, t, featurePointxy, lmk_face_idx, lmk_b_coords, lmk_facevtx_idx):

    R = rodrigues(pi)
    proj = np.array([[1,0,0], [0,1,0]])
#    selectedFacePoints = faceshape[list(fp_index), :]
    from fitting.util import mesh_points_by_barycentric_coordinates
    v_selected_3d = mesh_points_by_barycentric_coordinates(faceshape, mesh_face, lmk_face_idx, lmk_b_coords)
    use_lunkuo =1
    source_face_lmkvtx = faceshape[lmk_facevtx_idx[0:17]]
    if use_lunkuo:
        frame_landmark_idx = range(0,17)+range(17, 60) + range(61, 64) + range(65, 68)
    else:
        frame_landmark_idx = range(17, 60) + range(61, 64) + range(65, 68)
    if use_lunkuo:
        v_selected_merge = np.vstack([source_face_lmkvtx, v_selected_3d])
    else:
        v_selected_merge = v_selected_3d
    selectedFacePoints = v_selected_merge


    selectedFacePointxy = lamb * (proj.dot(R.dot(selectedFacePoints.T))) + t.reshape([2,1])
    errorMatrix = selectedFacePointxy - featurePointxy[frame_landmark_idx,:].T
    return LA.norm(errorMatrix, "fro")
def optRSTfromShape_t(faceshape,mesh_face, R0, s0, t0, featurePointxy, lmk_face_idx, lmk_b_coords, lmk_facevtx_idx, R_pre, s_pre, t_pre):
    global h_6, h_7, h_8
    pi_pre = rodrigues(R_pre).reshape(3)
    def energyFunc (x):
        pi = (x[0:3])
        lamb = x[3]
        t = x[4:6]
        return featureEnergyGivenshape(faceshape,mesh_face, pi, lamb, t, featurePointxy, lmk_face_idx, lmk_b_coords, lmk_facevtx_idx)  + h_6 * LA.norm(pi-pi_pre) + h_7 * LA.norm(lamb - s_pre) + h_8 * LA.norm(t - t_pre.reshape(2))
    x0 = np.zeros([6])
    x0[0:3] = rodrigues(R0).reshape(3)
    x0[3] = s0
    x0[4:6] = t0.reshape(2)
    x = optimize.fmin_bfgs(energyFunc, x0, maxiter = 50)
    R = rodrigues(x[0:3])
    s = x[3]
    t = x[4:6]
    return R, s, t

if __name__ == '__main__':
    faceshape =[]
    face = FaceModel(faceshape)
    face.compute_Laplacian()
    # generate current blendshapes
    face.generate_blendshape_from_b0(faceshape)

    beta_seq = []
    R_seq = []
    t_seq = []
    s =1
    beta_seq.append(np.zeros([1, face.n_expresses]))
    # R_seq.append(R)
    # t_seq.append(t)

    s_seq = []
    s_seq.append(s)
    featurePointxy = []
    fp_index = []
    from triangle_raster import BBoxi_2d
    #boundShape = np.sqrt((dets[0].left() - dets[0].right()) ** 2 + (dets[0].top() - dets[0].bottom()) ** 2)
    bbox2d = BBoxi_2d(featurePointxy)
    p1 = np.array([BBoxi_2d.min[0],BBoxi_2d.min[1]])
    p2 = np.array([BBoxi_2d.max[0],BBoxi_2d.max[1]])
    boundShape = np.linalg.norm(p1-p2)
    boundShape = 1. / boundShape
    R = []
    s = []
    t =[]
    beta = optexpressFromRST_t(face, R, s, t, featurePointxy, fp_index, beta_seq[0], boundShape,cur_frame_num = 0)
    faceshape = face.compute_shape_from_blendshape(beta)
    pass