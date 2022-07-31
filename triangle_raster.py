# -- coding: utf-8 --
import numpy as np
from numpy.linalg import norm as norm
from numba import jit, jitclass, int32, float32, float64
import math
from time import time


class BBoxf:
    def __init__(self, v0=np.array([]), v1=np.array([]), v2=np.array([])):
        self.min = np.array([100000.0, 100000.0])
        self.max = np.array([-100000.0, -100000.0])
        if v0.size > 0 and v1.size > 0 and v2.size > 0:
            t0 = [v0[0], v1[0], v2[0]]
            t1 = [v0[1], v1[1], v2[1]]
            self.min[0] = min(t0)
            self.min[1] = min(t1)
            self.max[0] = max(t0)
            self.max[1] = max(t1)


class BBox3f:
    def __init__(self, v0=np.array([]), v1=np.array([]), v2=np.array([])):
        self.min = np.array([100000.0, 100000.0, 100000.0])
        self.max = np.array([-100000.0, -100000.0, -100000.0])
        if v0.size > 0 and v1.size > 0 and v2.size > 0:
            t0 = [v0[0], v1[0], v2[0]]
            t1 = [v0[1], v1[1], v2[1]]
            t2 = [v0[2], v1[2], v2[2]]
            self.min[0] = min(t0)
            self.min[1] = min(t1)
            self.min[2] = min(t2)
            self.max[0] = max(t0)
            self.max[1] = max(t1)
            self.max[1] = max(t2)

    # N*3
    def addvertex_array(self, v_array):
        if v_array.shape[0] > 0 and v_array.shape[1] > 0:
            t0 = v_array[:, 0]
            t1 = v_array[:, 1]
            t2 = v_array[:, 2]
            self.min[0] = min(t0)
            self.min[1] = min(t1)
            self.min[2] = min(t2)
            self.max[0] = max(t0)
            self.max[1] = max(t1)
            self.max[2] = max(t2)


class BBoxi_2d:
    def __init__(self, v_array=np.array([])):
        self.min = np.array([100000.0, 100000.0, 100000.0])
        self.max = np.array([-100000.0, -100000.0, -100000.0])
        if v_array.size > 0:
            t0 = v_array[:, 0]
            t1 = v_array[:, 1]
            self.min[0] = min(t0)
            self.min[1] = min(t1)
            self.max[0] = max(t0)
            self.max[1] = max(t1)


class BBoxi:
    def __init__(self, v0=np.array([]), v1=np.array([]), v2=np.array([])):
        self.min = np.array([100000, 100000])
        self.max = np.array([-100000, -100000])
        if v0.size > 0 and v1.size > 0 and v2.size > 0:
            t0 = [v0[0], v1[0], v2[0]]
            t1 = [v0[1], v1[1], v2[1]]
            self.min[0] = min(t0)
            self.min[1] = min(t1)
            self.max[0] = max(t0)
            self.max[1] = max(t1)


class BBox3i:
    def __init__(self, v0=np.array([]), v1=np.array([]), v2=np.array([])):
        self.min = np.array([100000, 100000, 100000])
        self.max = np.array([-100000, -100000, -100000])
        if v0.size > 0 and v1.size > 0 and v2.size > 0:
            t0 = [v0[0], v1[0], v2[0]]
            t1 = [v0[1], v1[1], v2[1]]
            t2 = [v0[2], v1[2], v2[2]]
            self.min[0] = min(t0)
            self.min[1] = min(t1)
            self.min[2] = min(t2)
            self.max[0] = max(t0)
            self.max[1] = max(t1)
            self.max[2] = max(t2)


'''
  brief Auxiliairy data structure for computing face face adjacency information.
It identifies and edge storing two vertex pointer and a face pointer where it belong.
'''


@jit(nopython=True, cache=True)
def compare(pl, pr):
    if pl[0] < pr[0]:
        return -1
    elif pl[0] > pr[0]:
        return 1
    else:
        if pl[1] < pr[1]:
            return -1
        elif pl[1] == pr[1]:
            return 0
        else:
            return 1


@jit(nopython=True, cache=True)
def compare3D(pl, pr):
    if pl[0] < pr[0]:
        return -1
    elif pl[0] > pr[0]:
        return 1
    else:
        if pl[1] < pr[1]:
            return -1
        elif pl[1] > pr[1]:
            return 1
        else:
            if pl[2] < pr[2]:
                return -1
            elif pl[2] > pr[2]:
                return 1
            else:
                return 0


spec_PEdgeTex = [
    ('v', float64[:, :]),  # a simple scalar field
    ('f', int32),  # an array field
    ('z', int32),
]


# @jitclass(spec_PEdgeTex)
class PEdgeTex(object):

    def __init__(self):
        self.v = np.zeros((2, 2))  # the two TexCoord are ordered!
        self.f = -1  # the face where this edge belong
        self.z = -1  # index in [0..2] of the edge of the face

    def set(self, mesh, pf, nz):
        if pf == -1 or nz < 0 or nz > 2:
            print "PEdgeTex: set input error\n"
            return
        self.v[0, :] = mesh.wt(pf, nz)
        self.v[1, :] = mesh.wt(pf, (nz + 1) % 3)  # 假设是三角形连接
        # 确保顺序小到大
        if self.v[1, 0] < self.v[0, 0]:
            tmp = self.v[0, :].copy()
            self.v[0, :] = self.v[1, :].copy()
            self.v[1, :] = tmp
        elif self.v[1, 0] == self.v[0, 0]:
            if self.v[1, 1] < self.v[0, 1]:
                tmp = self.v[0, :].copy()
                self.v[0, :] = self.v[1, :].copy()
                self.v[1, :] = tmp
        self.f = pf
        self.z = nz

    def __cmp__(self, pe):
        if __lt__(pe) == True:
            return -1
        elif __eq__(pe) == True:
            return 0
        else:
            return 1

    def __lt__(self, pe):
        if compare(self.v[0, :], pe.v[0, :]) < 0:
            return True
        elif compare(self.v[0, :], pe.v[0, :]) > 0:
            return False
        else:
            if compare(self.v[1, :], pe.v[1, :]) < 0:
                return True
            else:
                return False

    def __eq__(self, pe):
        if compare(self.v[0, :], pe.v[0, :]) == 0 and compare(self.v[1, :], pe.v[1, :]) == 0:
            return True
        else:
            return False

    def __ne__(self, pe):
        self.no_equal(self, pe)

    def no_equal(self, pe):
        if compare(self.v[0, :], pe.v[0, :]) != 0 or compare(self.v[1, :], pe.v[1, :]) != 0:
            return True
        else:
            return False


class PEdge:
    def __init__(self):
        self.v = np.zeros((2, 3))  # the two vertex are ordered!
        self.f = -1  # the face where this edge belong
        self.z = -1  # index in [0..2] of the edge of the face

    def set(self, mesh, pf, nz):
        if pf == -1 or nz < 0 or nz > 2:
            print "PEdge: set input error\n"
            return
        self.v[0, :] = mesh.f_v(pf, nz)
        self.v[1, :] = mesh.f_v(pf, (nz + 1) % 3)  # 假设是三角形连接
        # 确保顺序小到大
        sign = compare3D(self.v[0, :], self.v[1, :])
        if sign > 0:
            tmp = self.v[0, :].copy()
            self.v[0, :] = self.v[1, :].copy()
            self.v[1, :] = tmp
        self.f = pf
        self.z = nz

    def __cmp__(self, pe):
        if __lt__(pe) == True:
            return -1
        elif __eq__(pe) == True:
            return 0
        else:
            return 1

    def __lt__(self, pe):
        if compare3D(self.v[0, :], pe.v[0, :]) < 0:
            return True
        elif compare3D(self.v[0, :], pe.v[0, :]) > 0:
            return False
        else:
            if compare3D(self.v[1, :], pe.v[1, :]) < 0:
                return True
            else:
                return False

    def __eq__(self, pe):
        if compare3D(self.v[0, :], pe.v[0, :]) == 0 and compare3D(self.v[1, :], pe.v[1, :]) == 0:
            return True
        else:
            return False

    def __ne__(self, pe):
        if compare3D(self.v[0, :], pe.v[0, :]) != 0 or compare3D(self.v[1, :], pe.v[1, :]) != 0:
            return True
        else:
            return False


spec_mesh = [
    ('v', float64[:, :]),  # a simple scalar field
    ('vertex_color', float64[:, :]),  # an array field
    ('normal', float64[:, :]),
    ('vt', float64[:, :]),
    ('face', int32[:, :]),
    ('n_face', int32[:, :]),
    ('t_face', int32[:, :]),
    ('FFp_', int32[:, :]),
    ('FFi_', int32[:, :]),
    ('FFP_V', int32[:, :]),
    ('FFP_V', int32[:, :]),

]


# @jitclass(spec_mesh)
class MetroMesh(object):

    def __init__(self):
        # self.v = np.array([],dtype=np.float64)  # np.array((N,3),dtype= float)
        # self.vertex_color = np.array([0.0],dtype=np.float64)
        # self.normal = np.array([0.0],dtype=np.float64)  # np.array((N,3),dtype= float)
        # self.vt = np.array([0],dtype=np.float64)  # np.array((N,2),dtype= float)
        # self.face = np.array([0],dtype=np.int32)  # np.array((N,9),dtype= int) vf,tf,nf
        # self.n_face = np.array([0],dtype=np.int32)
        # self.t_face = np.array([0],dtype=np.int32)
        # self.FFp_ = np.array([0],dtype=np.int32) #use for texture face
        # self.FFi_ = np.array([0],dtype=np.int32) #use for texture face
        # self.FFP_V=np.array([0],dtype=np.int32) # use for vertex face
        # self.FFP_V = np.array([0],dtype=np.int32) #use for vertex face
        pass

    def set_mesh(self, v, vertex_color=np.array([]), normal=np.array([]), vt=np.array([]), face=np.array([]),
                 n_face=np.array([]), t_face=np.array([])):
        self.v = v.copy()
        self.vertex_color = vertex_color.copy()
        self.normal = normal.copy()
        self.vt = vt.copy()
        self.face = face.copy()
        self.n_face = n_face.copy()
        self.t_face = t_face.copy()

    # 第 face_id 个面的，纹理连接关系中的第id个uv点的uv值
    def wt(self, face_id, id):
        vt_id = self.t_face[face_id, id]
        return self.vt[vt_id, :]

    def f_v(self, face_id, id):
        v_id = self.face[face_id, id]
        return self.v[v_id, :]

    def v_color(self, face_id, id):
        vt_id = self.face[face_id, id]
        return self.vertex_color[vt_id, :]

    def num_texture_face(self):
        return self.t_face.shape[0]

    #    jit(nopython=True, cache=True)
    def FaceFaceFromTexCoord(self):
        if self.vt.size <= 0 or self.t_face.size <= 0:
            return
        # 假设连接关系都是三角形
        # n_edges =np.array((self.t_face.shape[0]*3,1),dtype= PEdgeTex)
        n_edges = []
        face_num = self.t_face.shape[0]
        for pf in range(0, face_num):
            for j in range(0, 3):
                if compare(self.wt(pf, j), self.wt(pf, (j + 1) % 3)) != 0:  # 一个面上的两个点的纹理坐标不应该相同
                    edge = PEdgeTex()
                    edge.set(self, pf, j)
                    n_edges.append(edge)
        # edge 排序？
        n_edges.sort()
        edge_len = len(n_edges)
        ps = 0
        pe = 0
        ne = 0
        self.FFp_ = np.zeros((face_num, 3), dtype=np.int32)
        self.FFi_ = np.zeros((face_num, 3), dtype=np.int32)
        while 1:
            if pe == edge_len or n_edges[pe].no_equal(n_edges[ps]):
                q = 0
                for i in range(ps, pe - 1):
                    q = i
                    q_next = q + 1
                    self.FFp_[n_edges[q].f, n_edges[q].z] = n_edges[q_next].f
                    self.FFi_[n_edges[q].f, n_edges[q].z] = n_edges[q_next].z
                if ps < pe - 1:  # 这里要注意
                    q = pe - 1
                else:
                    q = ps
                self.FFp_[n_edges[q].f, n_edges[q].z] = n_edges[ps].f
                self.FFi_[n_edges[q].f, n_edges[q].z] = n_edges[ps].z
                ps = pe
                ne += 1
            if pe == edge_len:
                break
            pe += 1

    def IsBorder(self, f, j):
        if self.FFp_.size > 0:  # FaceType::HasFFAdjacency()
            if f < self.FFp_.size:
                return self.FFp_[f, j] == f  # 如果一个面的一条边的邻接面是自己，则这条边是边界边

    def FaceFace(self):
        if self.v.size <= 0 or self.face.size <= 0:
            return
        # 假设连接关系都是三角形
        # n_edges =np.array((self.t_face.shape[0]*3,1),dtype= PEdgeTex)
        n_edges = []
        face_num = self.face.shape[0]
        for pf in range(0, face_num):
            for j in range(0, 3):
                if compare3D(self.f_v(pf, j), self.f_v(pf, (j + 1) % 3)) != 0:  # 一个面上的两个点的纹理坐标不应该相同
                    edge = PEdge()
                    edge.set(self, pf, j)
                    n_edges.append(edge)
        # edge 排序？
        n_edges.sort()
        edge_len = len(n_edges)
        ps = 0
        pe = 0
        ne = 0
        self.FFp_V = np.zeros((face_num, 3), dtype=int)
        self.FFi_V = np.zeros((face_num, 3), dtype=int)
        while 1:
            if pe == edge_len or n_edges[pe] != n_edges[ps]:
                q = 0
                for i in range(ps, pe - 1):
                    q = i
                    q_next = q + 1
                    self.FFp_V[n_edges[q].f, n_edges[q].z] = n_edges[q_next].f
                    self.FFi_V[n_edges[q].f, n_edges[q].z] = n_edges[q_next].z
                if ps < pe - 1:  # 这里要注意
                    q = pe - 1
                else:
                    q = ps
                self.FFp_V[n_edges[q].f, n_edges[q].z] = n_edges[ps].f
                self.FFi_V[n_edges[q].f, n_edges[q].z] = n_edges[ps].z
                ps = pe
                ne += 1
            if pe == edge_len:
                break
            pe += 1

    def IsBorder_V(self, f, j):
        if self.FFp_V.size > 0:  # FaceType::HasFFAdjacency()
            if f < self.FFp_V.size:
                return self.FFp_V[f, j] == f  # 如果一个面的一条边的邻接面是自己，则这条边是边界边


class VertexSampler:
    def __init__(self, _srcMesh, _srcImg, upperBound):
        self.srcMesh = _srcMesh
        self.srcImg = _srcImg
        self.upperBound = upperBound

        def unifGridFace(mesh):
            bbox = BBox3f()
            vertex_array = []
            for i in range(0, mesh.v.shape[0]):
                vertex_array.append(mesh.v[i, :])
            vertex_array = np.array(vertex_array)
            bbox.addvertex_array(vertex_array)

    def getClosetFace(self):
        pass

    def InterpolationParameters(self, nearestFid, mesh, Nomral, closestPt):
        def InterpolationParameters_inter(nearestFid, mesh, idx, closestPt):
            v0 = mesh.v[mesh.v_f[nearestFid, 0]]
            v1 = mesh.v[mesh.v_f[nearestFid, 1]]
            v2 = mesh.v[mesh.v_f[nearestFid, 2]]
            P = np.zeros((3, 2))
            bq = []
            if idx == 0:
                P[0, :] = [v0[1], v0[2]]
                P[1, :] = [v1[1], v0[2]]
                P[2, :] = [v2[1], v0[2]]
                bq.append([closestPt[1], closestPt[2]])
            elif idx == 1:
                P[0, :] = [v0[0], v0[2]]
                P[1, :] = [v1[0], v0[2]]
                P[2, :] = [v2[0], v0[2]]
                bq.append([closestPt[0], closestPt[2]])
            else:
                P[0, :] = [v0[0], v0[1]]
                P[1, :] = [v1[0], v0[1]]
                P[2, :] = [v2[0], v0[1]]
                bq.append([closestPt[0], closestPt[1]])
            EPSILON = 0.0001

            x1 = P[0, 0]
            x2 = P[1, 0]
            x3 = P[2, 0]
            y1 = P[0, 1]
            y2 = P[1, 1]
            y3 = P[2, 1]
            x = bq[0]
            y = bq[1]
            L1 = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / ((y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3))
            L2 = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / ((y3 - y1) * (x2 - x3) + (x1 - x3) * (y2 - y3))
            L3 = 1 - L1 - L2
            if math.isnan(L1) or math.isnan(L2) or math.isnan(L3):
                print 'occur nan\n'
                L1 = L2 = L3 = 1.0 / 3.0
            inside = True
            inside &= (L1 >= 0 - EPSILON) and (L1 <= 1 + EPSILON)
            inside &= (L2 >= 0 - EPSILON) and (L2 <= 1 + EPSILON)
            inside &= (L3 >= 0 - EPSILON) and (L3 <= 1 + EPSILON)
            return inside, L1, L2, L3

        if abs(Nomral[0]) > abs(Nomral[1]):
            if abs(Nomral[0]) > abs(Nomral[2]):
                InterpolationParameters_inter(nearestFid, mesh, 0, closestPt)
            else:
                InterpolationParameters_inter(nearestFid, mesh, 2, closestPt)
        else:
            if abs(Nomral[1]) > abs(Nomral[2]):
                InterpolationParameters_inter(nearestFid, mesh, 1, closestPt)
            else:
                InterpolationParameters_inter(nearestFid, mesh, 2, closestPt)

    def AddVert(self, target_mesh, target_mesh_v_id):
        closestPt = np.zeros((1, 3))
        dist = self.dist_upper_bound
        nearestF, dist = self.getClosetFace()
        Nomral = []
        interp = self.InterpolationParameters(nearestF, self.srcMesh, Nomral, closestPt)
        w = self.srcImg.shape[1]
        h = self.srcImg.shape[0]
        interp = interp.reshape(1, 3)
        texture_cood = []
        for i in range(0, 3):
            uv = self.srcMesh.wt(nearestF, i)
            texture_cood.append([uv[0], uv[1]])
        texture_cood = np.array(texture_cood)
        interp_uv = np.dot(interp, texture_cood)
        x = w * interp_uv[0]
        y = h * (1.0 - interp_uv[1])
        # repeat mode
        x = (x % w + w) % w
        y = (y % h + h) % h
        bgr = self.srcImg[y, x]
        target_mesh.v_color[target_mesh_v_id] = [bgr[2], bgr[1], bgr[0]]  # convert to rgb


g_count = 0


class RasterSampler:

    def __init__(self, image):
        self.image = image
        pass

    # expects points outside face (affecting face color) with edge distance > 0
    # f Nx3 ,p 重心坐标， tp纹理坐标
    def AddTextureSample(self, mesh, f_id, p, tp, edgeDist=0.0):
        c = []  # 重心坐标对应的颜色值
        alpha = 255
        if edgeDist != 0.0:
            alpha = int(254 - edgeDist * 128)
        FLT_MAX = 3.40282346638528860e+38
        if edgeDist == FLT_MAX:
            print FLT_MAX
        height, width, dim = self.image.shape
        if alpha == 255 or self.image[height - 1 - tp[1], tp[0], 3] < alpha:
            v0_color = mesh.v_color(f_id, 0)
            v1_color = mesh.v_color(f_id, 1)
            v2_color = mesh.v_color(f_id, 2)
            v0_color = v0_color.reshape(v0_color.size, 1)
            v1_color = v1_color.reshape(v1_color.size, 1)
            v2_color = v2_color.reshape(v2_color.size, 1)
            m = np.hstack([v0_color, v1_color, v2_color])
            #    print p
            p = p.reshape(p.size, 1)
            c = np.dot(m, p)
            self.image[height - 1 - tp[1], tp[0], :] = [c[2], c[1], c[0], alpha]  # BGRA 这里要想想为什么height -1


@jit(nopython=True, cache=True)
def ClosestPoint(borderEdges, px):
    if borderEdges.size == 4:
        p0 = borderEdges[0:2]
        p1 = borderEdges[2:4]
    elif borderEdges.size == 6:
        p0 = borderEdges[0:2]
        p1 = borderEdges[3:5]
    dir = (p1 - p0) / norm(p1 - p0)
    t = np.dot(px - p0, dir)  # /norm(px-p0)
    length = norm(p1 - p0)
    if t <= 0:
        return p0
    elif t > length:
        return p1
    else:
        return p0 + dir * t


# v0 2x1 v1 2x1 v3 2x1


def SingleFaceRaster(mesh, f, ps, v0, v1, v2, correctSafePointsBaryCoords):
    FLT_MAX = 3.40282346638528860e+38

    bboxf = BBoxf(v0, v1, v2)
    bbox = BBoxi()
    bbox.min[0] = math.floor(bboxf.min[0])
    bbox.min[1] = math.floor(bboxf.min[1])
    bbox.max[0] = math.ceil(bboxf.max[0])
    bbox.max[1] = math.ceil(bboxf.max[1])
    # Calcolo versori degli spigoli ,the preparation products scalari
    d10 = v1 - v0
    d21 = v2 - v1
    d02 = v0 - v2
    # Preparazione prodotti scalari
    b0 = (bbox.min[0] - v0[0]) * d10[1] - (bbox.min[1] - v0[1]) * d10[0]
    b1 = (bbox.min[0] - v1[0]) * d21[1] - (bbox.min[1] - v1[1]) * d21[0]
    b2 = (bbox.min[0] - v2[0]) * d02[1] - (bbox.min[1] - v2[1]) * d02[0]
    # Preparazione degli steps
    db0 = d10[1]
    db1 = d21[1]
    db2 = d02[1]
    # Preparazione segni
    dn0 = -d10[0]
    dn1 = -d21[0]
    dn2 = -d02[0]
    # Calculating orientation
    flipped = bool(not (d02.dot(np.array([-d10[1], d10[0]])) >= 0))
    # Calculating border edges
    borderEdges = np.zeros((3, 4))
    edgeLength = np.array([0.0, 0.0, 0.0])
    edgeMask = 0

    def edge_length(edge):
        v0 = edge[0:2]
        v1 = edge[2:4]
        return norm(v0 - v1)

    if (mesh.IsBorder(f, 0)):
        borderEdges[0, :] = np.hstack([v0, v1])
        edgeLength[0] = edge_length(borderEdges[0, :])
        edgeMask |= 1

    if (mesh.IsBorder(f, 1)):
        borderEdges[1, :] = np.hstack([v1, v2])
        edgeLength[1] = edge_length(borderEdges[1, :])
        edgeMask |= 2

    if (mesh.IsBorder(f, 2)):
        borderEdges[2, :] = np.hstack([v2, v0])
        edgeLength[2] = edge_length(borderEdges[2, :])
        edgeMask |= 4
    # Rasterizzazione
    #  | 1 xa ya |
    #  | 1 xa ya |
    #  | 1 xc yc |   * 0.5 即为 三角形a，b,c面积公式
    de = v0[0] * v1[1] - v0[0] * v2[1] - v1[0] * v0[1] + v1[0] * v2[1] - v2[0] * v1[1] + v2[0] * v0[1]

    for x in range(bbox.min[0] - 1, bbox.max[0] + 1 + 1):
        bool_in = False
        n = [b0 - db0 - dn0, b1 - db1 - dn1, b2 - db2 - dn2]
        for y in range(bbox.min[1] - 1, bbox.max[1] + 1 + 1):

            if (((n[0] >= 0 and n[1] >= 0 and n[2] >= 0) or (n[0] <= 0 and n[1] <= 0 and n[2] <= 0)) and (de != 0)):

                baryCoord = np.array([0.0, 0.0, 0.0])
                baryCoord[0] = np.float(
                    -y * v1[0] + v2[0] * y + v1[1] * x - v2[0] * v1[1] + v1[0] * v2[1] - x * v2[1]) / de
                baryCoord[1] = -np.float(
                    x * v0[1] - x * v2[1] - v0[0] * y + v0[0] * v2[1] - v2[0] * v0[1] + v2[0] * y) / de
                baryCoord[2] = 1 - baryCoord[0] - baryCoord[1]

                ps.AddTextureSample(mesh, f, baryCoord, [x, y], 0)
                bool_in = True
            else:
                # Check whether a pixel outside (on a border edge side) triangle affects color inside it
                px = np.array([float(x), float(y)])
                closePoint = np.array([0.0, 0.0])
                closeEdge = -1
                minDst = FLT_MAX

                # find the closest point (on some edge) that lies on the 2x2 squared neighborhood of the considered point
                for i in range(0, 3):
                    if (edgeMask & (1 << i)):
                        if (((not flipped) and (n[i] < 0)) or
                                (flipped and (n[i] > 0))):
                            close = ClosestPoint(borderEdges[i, :], px)
                            dst = norm(close - px)
                            if (dst < minDst and
                                    close[0] > px[0] - 1 and close[0] < px[0] + 1 and
                                    close[1] > px[1] - 1 and close[1] < px[1] + 1):
                                minDst = dst
                                closePoint = close
                                closeEdge = i

                if (closeEdge >= 0):
                    baryCoord = np.array([0.0, 0.0, 0.0])
                    if (correctSafePointsBaryCoords):
                        # Add x,y sample with closePoint barycentric coords (on edge)
                        baryCoord[closeEdge] = norm(closePoint - borderEdges[closeEdge, 2:4]) / edgeLength[closeEdge]
                        baryCoord[(closeEdge + 1) % 3] = 1 - baryCoord[closeEdge]
                        baryCoord[(closeEdge + 2) % 3] = 0
                    else:
                        # Add x,y sample with his own barycentric coords (off edge)
                        baryCoord[0] = np.float(
                            -y * v1[0] + v2[0] * y + v1[1] * x - v2[0] * v1[1] + v1[0] * v2[1] - x * v2[1]) / de
                        baryCoord[1] = - np.float(
                            x * v0[1] - x * v2[1] - v0[0] * y + v0[0] * v2[1] - v2[0] * v0[1] + v2[0] * y) / de
                        baryCoord[2] = 1 - baryCoord[0] - baryCoord[1]

                    ps.AddTextureSample(mesh, f, baryCoord, [x, y], minDst)
                    bool_in = True

            n[0] += dn0
            n[1] += dn1
            n[2] += dn2

        b0 += db0
        b1 += db1
        b2 += db2


# MetroMesh & m VertexSampler &ps

def Texture(m, ps, textureWidth, textureHeight, correctSafePointsBaryCoords=True):
    print "Similar Triangles face sampling\n"

    for fi in range(0, m.face.shape[0]):
        ti = np.zeros([3, 2])
        for i in range(0, 3):
            ti[i] = np.array([m.wt(fi, i)[0] * textureWidth - 0.5, m.wt(fi, i)[1] * textureHeight - 0.5])
        SingleFaceRaster(m, fi, ps, ti[0, :], ti[1, :], ti[2, :], correctSafePointsBaryCoords)


def VertexUniform(target_mesh, vs, sampleNum):
    vtx_num = target_mesh.v.shape[0]
    if sampleNum > vtx_num:
        for i in range(0, vtx_num):
            vs.AddVert(target_mesh=target_mesh, target_mesh_v_id=i)
        return

    def FillAndShuffleVertexPointerVector(list):
        import random
        random.shuffle(list)
        return list

    list = range(0, vtx_num)
    list = FillAndShuffleVertexPointerVector(list)
    for i in range(0, sampleNum):
        vs.AddVert(target_mesh=target_mesh, target_mesh_v_id=list[i])


def print_para_resut(step, timer_end, timer_start):
    print step
    print "in %f sec\n" % (timer_end - timer_start)


def FP_COLOR_TO_TEXTURE(file_path, mesh, textW, textH, overwrite=True, assign=True, pp=True):
    import cv2

    image = np.zeros((textH, textW, 4), np.uint8)  # BGRA
    image[:, :, :] = [0, 0, 0, 0]
    # 建立 vt 的连接关系
    # 0 代表 v0-v1 , 1 代表 v1-v2 ,2 代表 v1-v2
    #    border_flags = np.zeros( mesh.num_texture_face,3)

    #  Compute (texture-space) border edges

    timer_start = time()
    mesh.FaceFaceFromTexCoord()
    timer_end = time()
    print_para_resut('FaceFaceFromTexCoord()', timer_end, timer_start)

    timer_start = time()
    ps = RasterSampler(image)
    Texture(mesh, ps, textW, textH, True)
    timer_end = time()
    print_para_resut('Texture()', timer_end, timer_start)
    count = 0
    for hieght in range(0, textH):
        for width in range(0, textW):
            pixel = image[hieght, width, :]
            if pixel[3] < 255 and (not pp or pixel[3]) > 0:
                #                print pixel[3]
                pixel[3] = 255
                count += 1
                if pixel[3] < 255:
                    pixel[3] = 255
    if pp:
        timer_start = time()
        PullPush(image, np.array([0, 0, 0, 0]))
        timer_end = time()
        print_para_resut('pp()', timer_end, timer_start)
    cv2.imwrite(file_path, image)


# @jit(nopython=True, cache=True)
def PullPush(p, bkcolor):
    #    print(p.shape)
    # list_2d = [[0 for col in range(cols)] for row in range(rows)]

    mip = []  # [0 for row in range(16)]
    for i in range(0, 16):
        mip.append(0)
    div = 2
    miplev = 0
    # pull phase create the mipmap
    #    timer_start = time()
    while 1:
        img = np.zeros((p.shape[0] / div, p.shape[1] / div, 4), np.uint8)
        img[:, :, :] = bkcolor
        mip[miplev] = img
        div *= 2
        if miplev > 0:
            PullPushMip(mip[miplev - 1], mip[miplev], bkcolor)
        else:
            PullPushMip(p, mip[miplev], bkcolor)

        # if 0:
        #     cv2.imwrite('D:/mprojects/flame-fitting/uv_mapping/bwm/test_simplication/'
        #             +'mip_'+str(miplev)+'.png'
        #             ,  mip[miplev])
        #        print(mip[miplev])
        #        @jit(nopython=True, cache=True)
        def stop_con(p):
            if p.shape[1] <= 4 or p.shape[0] <= 4:
                return True
            else:
                False

        if stop_con(mip[miplev]):  # mip[miplev].shape[1] <= 4 or mip[miplev].shape[0] <= 4:
            break
        miplev += 1
    miplev += 1
    #    timer_end = time()
    #    print_para_resut('PullPushMip()', timer_end, timer_start)
    #    timer_start = time()
    for i in range(miplev - 1, -1, -1):
        if i:
            #            print(mip[i-1])
            PullPushFill(mip[i - 1], mip[i], bkcolor)
        else:
            #            print(p)
            PullPushFill(p, mip[i], bkcolor)
        # if 0:
        #     cv2.imwrite('D:/mprojects/flame-fitting/uv_mapping/bwm/test_simplication/'
        #             +'mipfill_'+str(i)+'.png'
        #             ,  mip[i])


#    timer_end = time()
#    print_para_resut('PullPushFill()', timer_end, timer_start)
@jit(nopython=True, cache=True)
def mean4w(p1, w1, p2, w2, p3, w3, p4, w4):
    result = (p1 * int(w1) + p2 * int(w2) + p3 * int(w3) + p4 * int(w4)) / (int(w1) + int(w2) + int(w3) + int(w4))
    return int(result)


@jit(nopython=True, cache=True)
def mean4Pixelw(p1, w1, p2, w2, p3, w3, p4, w4):
    a = mean4w(p1[0], w1, p2[0], w2, p3[0], w3, p4[0], w4)
    b = mean4w(p1[1], w1, p2[1], w2, p3[1], w3, p4[1], w4)
    c = mean4w(p1[2], w1, p2[2], w2, p3[2], w3, p4[2], w4)
    d = mean4w(p1[3], w1, p2[3], w2, p3[3], w3, p4[3], w4)
    return np.array([a, b, c, d])


@jit(nopython=True, cache=True)
def ifequal(a, b):
    for i in range(len(a)):
        if a[i] != b[i]:
            return False
    return True


# @jit(nopython=True, cache=True)
def PullPushMip(p, mip, bkcolor):
    if p.shape[0] / 2 != mip.shape[0] or p.shape[1] / 2 != mip.shape[1]:
        print ('PullPushMip error')
        return

    for y in range(0, mip.shape[0]):
        for x in range(0, mip.shape[1]):
            if ifequal(p[y * 2, x * 2], bkcolor):
                w1 = 0
            else:
                w1 = 255
            if ifequal(p[y * 2, x * 2 + 1], bkcolor):
                w2 = 0
            else:
                w2 = 255
            if ifequal(p[y * 2 + 1, x * 2], bkcolor):
                w3 = 0
            else:
                w3 = 255
            if ifequal(p[y * 2 + 1, x * 2 + 1], bkcolor):
                w4 = 0
            else:
                w4 = 255
            if w1 + w2 + w3 + w4 > 0:
                mip[y, x] = mean4Pixelw(p[y * 2, x * 2, :], w1,
                                        p[y * 2, x * 2 + 1, :], w2,
                                        p[y * 2 + 1, x * 2, :], w3,
                                        p[y * 2 + 1, x * 2 + 1, :], w4)


# @jit(nopython=True, cache=True)
def PullPushFill(p, mip, bkg):
    #    print(p.shape)
    if p.shape[0] / 2 != mip.shape[0] or p.shape[1] / 2 != mip.shape[1]:
        print ('PullPushMip error')
        return
    for y in range(0, mip.shape[0]):
        for x in range(0, mip.shape[1]):
            if ifequal(p[y * 2, x * 2], bkg):
                p[y * 2, x * 2] = mean4Pixelw(mip[y, x, :], 144,
                                              mip[y, x - 1] if x > 0 else bkg, 48 if x > 0 else 0,
                                              mip[y - 1, x] if y > 0 else bkg, 48 if y > 0 else 0,
                                              mip[y - 1, x - 1] if x > 0 and y > 0 else bkg,
                                              16 if x > 0 and y > 0 else 0)
            if ifequal(p[y * 2, x * 2 + 1], bkg):
                p[y * 2, x * 2 + 1] = mean4Pixelw(mip[y, x, :], 144,
                                                  mip[y, x + 1] if x < mip.shape[1] - 1 else bkg,
                                                  48 if x < mip.shape[1] - 1 else 0,
                                                  mip[y - 1, x] if y > 0 else bkg, 48 if y > 0 else 0,
                                                  mip[y - 1, x + 1] if x < mip.shape[1] - 1 and y > 0 else bkg,
                                                  16 if x < mip.shape[1] - 1 and y > 0 else 0)
            if ifequal(p[y * 2 + 1, x * 2], bkg):
                p[y * 2 + 1, x * 2] = mean4Pixelw(mip[y, x, :], 144,
                                                  mip[y, x - 1] if x > 0 else bkg, 48 if x > 0 else 0,
                                                  mip[y + 1, x] if y < mip.shape[0] - 1 else bkg,
                                                  48 if y < mip.shape[0] - 1 else 0,
                                                  mip[y + 1, x - 1] if x > 0 and y < mip.shape[0] - 1 else bkg,
                                                  16 if x > 0 and y < mip.shape[0] - 1 else 0)
            if ifequal(p[y * 2 + 1, x * 2 + 1], bkg):
                p[y * 2 + 1, x * 2 + 1] = mean4Pixelw(mip[y, x, :], 144,
                                                      mip[y, x + 1] if x < mip.shape[1] - 1 else bkg,
                                                      48 if x < mip.shape[1] - 1 else 0,
                                                      mip[y + 1, x] if y < mip.shape[0] - 1 else bkg,
                                                      48 if y < mip.shape[0] - 1 else 0,
                                                      mip[y + 1, x + 1] if x < mip.shape[1] - 1 and y < mip.shape[
                                                          0] - 1 else bkg,
                                                      16 if x < mip.shape[1] - 1 and y < mip.shape[0] - 1 else 0)


def FP_TEX_TO_VCOLOR_TRANSFER(source_mesh, target_mesh, texture_path):
    import cv2
    source_image = cv2.imread(texture_path, cv2.IMREAD_UNCHANGED)
    upperbound = []
    vs = VertexSampler(source_mesh, source_image, upperbound)
    VertexUniform(target_mesh, vs, target_mesh.v.shape[0])
