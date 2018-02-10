# -- coding: utf-8 --

import numpy as np
import scipy as sp
import scipy.linalg as linalg
import networkx as nx


class Clustert1:
    '''
    http://blog.csdn.net/waleking/article/details/7584084
    '''
    def __init__(self):
        pass
    def getNormLaplacian(self,W):
        """input matrix W=(w_ij)
        "compute D=diag(d1,...dn)
        "and L=D-W
        "and Lbar=D^(-1/2)LD^(-1/2)
        "return Lbar
        """
        d = [np.sum(row) for row in W]
        D = np.diag(d)
        L = D - W
        # Dn=D^(-1/2)
        Dn = np.power(np.linalg.matrix_power(D, -1), 0.5)
        Lbar = np.dot(np.dot(Dn, L), Dn)
        return Lbar

    def getKSmallestEigVec(self,Lbar, k):
        """input
        "matrix Lbar and k
        "return
        "k smallest eigen values and their corresponding eigen vectors
        """
        eigval, eigvec = linalg.eig(Lbar)
        dim = len(eigval)

        # 查找前k小的eigval
        dictEigval = dict(zip(eigval, range(0, dim)))
        kEig = np.sort(eigval)[0:k]
        ix = [dictEigval[k] for k in kEig]
        return eigval[ix], eigvec[:, ix]

    def checkResult(self,Lbar, eigvec, eigval, k):
        """
        "input
        "matrix Lbar and k eig values and k eig vectors
        "print norm(Lbar*eigvec[:,i]-lamda[i]*eigvec[:,i])
        """
        check = [np.dot(Lbar, eigvec[:, i]) - eigval[i] * eigvec[:, i] for i in range(0, k)]
        length = [np.linalg.norm(e) for e in check] / np.spacing(1)
        print("Lbar*v-lamda*v are %s*%s" % (length, np.spacing(1)))

    def run(self):
        g = nx.karate_club_graph()
        nodeNum = len(g.nodes())
        m = nx.to_numpy_matrix(g)
        Lbar = self.getNormLaplacian(m)
        k = 2
        kEigVal, kEigVec = self.getKSmallestEigVec(Lbar, k)
        print("k eig val are %s" % kEigVal)
        print("k eig vec are %s" % kEigVec)
        self.checkResult(Lbar, kEigVec, kEigVal, k)

        # 跳过k means，用最简单的符号判别的方法来求点的归属

        clusterA = [i for i in range(0, nodeNum) if kEigVec[i, 1] > 0]
        clusterB = [i for i in range(0, nodeNum) if kEigVec[i, 1] < 0]

        # draw graph
        colList = dict.fromkeys(g.nodes())
        for node, score in colList.items():
            if node in clusterA:
                colList[node] = 0
            else:
                colList[node] = 0.6
        plt.figure(figsize=(8, 8))
        pos = nx.spring_layout(g)
        nx.draw_networkx_edges(g, pos, alpha=0.4)
        nx.draw_networkx_nodes(g, pos, nodelist=colList.keys(),
                               node_color=colList.values(),
                               cmap=plt.cm.Reds_r)
        nx.draw_networkx_labels(g, pos, font_size=10, font_family='sans-serif')
        import matplotlib.pyplot as plt
        plt.axis('off')
        plt.title("karate_club spectral clustering")
        plt.savefig("spectral_clustering_result.png")
        plt.show()

class Clustert2:
    '''
    http://www.dataivy.cn/blog/%E8%B0%B1%E8%81%9A%E7%B1%BBspectral-clustering/
    '''
    def __init__(self):
        pass
    def run(self):

        from sklearn.feature_extraction import image
        from sklearn.cluster import spectral_clustering

        # 生成原始图片信息
        l = 100
        x, y = np.indices((l, l))

        center1 = (28, 24)
        center2 = (40, 50)
        center3 = (77, 58)

        radius1, radius2, radius3 = 16, 14, 15

        circle1 = (x - center1[0]) ** 2 + (y - center1[1]) ** 2 < radius1 ** 2
        circle2 = (x - center2[0]) ** 2 + (y - center2[1]) ** 2 < radius2 ** 2
        circle3 = (x - center3[0]) ** 2 + (y - center3[1]) ** 2 < radius3 ** 2

        # 生成包括3个圆的图片
        img = circle1 + circle2 + circle3
        mask = img.astype(bool)
        img = img.astype(float)

        img += 1 + 0.2 * np.random.randn(*img.shape)
        graph = image.img_to_graph(img)
        graph.data = np.exp(-graph.data / graph.data.std())

        # 聚类输出
        labels = spectral_clustering(graph, n_clusters=4)
        label_im = -np.ones(mask.shape)
        #label_im[mask] = labels
        label_im  = np.reshape(labels ,(100,100))
        import matplotlib.pyplot as plt
        plt.matshow(img)
        plt.matshow(label_im)
        plt.show()

from sklearn.cluster import spectral_clustering
from  scipy.sparse import coo_matrix,dok_matrix
import scipy
from strand_convert import read_bin
class Strand_cluster:

    def __init__(self):
        pass
    def caculate_distance(selfs,strand1,strand2):

        strand1_len = 0.0
        strand2_len = 0.0
        strand1_root = strand1[0][:]
        strand2_root = strand2[0][:]
        for i in range(0,len(strand1)-1):
            tangent = strand1[i + 1][:] - strand1[i][:]
            strand1_len+=np.linalg.norm(tangent)
        for i in range(0,len(strand2)-1):
            tangent = strand2[i + 1][:] - strand2[i][:]
            strand2_len+=np.linalg.norm(tangent)
        #计算长度的距离
        stranLen_dis = np.linalg.norm(strand1_len-strand2_len)

        #计算root的距离
        root_dis = np.linalg.norm(strand1_root-strand2_root)
        #计算点到点距离
        p2p_dis =0.0
        sample_num = min( [len(strand1),len(strand2)])
        for i in range(0,sample_num):
            p2p_dis+= np.linalg.norm((strand1[i]-strand2[i]))
        #计算梯度距离
        gradient_dis  =0.0
        for i in range(0,sample_num-1):
            tangent1 = strand1[i + 1][:] - strand1[i][:]
            tangent2 = strand2[i + 1][:] - strand2[i][:]
            tangent1 /= np.linalg.norm(tangent1)
            tangent2 /= np.linalg.norm(tangent2)
            gradient_dis +=np.linalg.norm(tangent2-tangent1)
        # 计算拉普拉斯距离
        laplcian_dis = 0.0
        for i in range(1,sample_num-1):
            laplcian1 = 0.5*(strand1[i -1][:] + strand1[i+1][:]) - strand1[i][:]
            laplcian2 = 0.5 *(strand2[i - 1][:] + strand2[i + 1][:]) - strand2[i][:]
            laplcian_dis +=np.linalg.norm(laplcian2-laplcian1)

        all_dis = stranLen_dis+root_dis+p2p_dis+gradient_dis+laplcian_dis
        return all_dis

    def construct_affinite_matrix(self,result):
        strand_num = len(result)
        distance_matrix = dok_matrix((strand_num, strand_num))
        #构造上三角矩阵
        for i in range(0,strand_num):
            for j in range(i,strand_num):
                print i,j
                distance = self.caculate_distance(np.array(result[i]),np.array(result[j]))
                if distance <0:
                    print 'error strand distance'
                distance_matrix[i, j] = distance
        #因为距离矩阵是对称的，因此用下面的方式构造
        distance_matrix = distance_matrix + distance_matrix.T - scipy.sparse.diags(distance_matrix.diagonal())

        # Take a decreasing function of the gradient: an exponential
        # The smaller beta is, the more independent the segmentation is of the
        # actual image. For beta=1, the segmentation is close to a voronoi
        beta = 5
        eps = 1e-6
        distance_matrix.data = np.exp(-beta * distance_matrix.data / distance_matrix.data.std()) + eps #转化为相似性矩阵
        return distance_matrix
    def construct_feature(self,result,select_strand_id):
        feature_num = 10
        feature_matrix = dok_matrix(len(select_strand_id), feature_num)
        for i,strand_id in enumerate(select_strand_id):
            feature_matrix[i,0] =0
        return  feature_matrix


    def run(self,prj_dir,file_name):

        result = read_bin(prj_dir + file_name + '.data')
        #筛选出长度太小的
        select_strand_id =[]
        select_strand = []
        for i in range(0, result.shape[0], 1):
            strand = result[i]
            if len(strand) > 2:
                select_strand_id.append(i)
            select_strand.append(strand)

        #labels = spectral_clustering(strand_graph, n_clusters=10)
        #shape(n_samples, n_features)
        #featrue_vec = self.construct_feature(result,select_strand_id)
        #sc = spectral_clustering( n_clusters=10)
        #sc.fit(featrue_vec)
        # print sc.labels_
        strand_graph = self.construct_affinite_matrix(select_strand)
        print 'construct sucess'
        label = spectral_clustering(strand_graph,n_clusters=10)
        print label


#c1 = Clustert1()
#c1.run()
#c2 = Clustert2()
#c2.run()
prj_dir = 'G:/yuanqing/faceproject/hairstyles/hairstyles/hair/'
sc = Strand_cluster()
sc.run(prj_dir,'strands00002')

