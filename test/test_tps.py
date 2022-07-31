# -- coding: utf-8 --
import ipdb
import numpy as np
import numpy.linalg as nl
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, cdist, squareform
from numba import jit,jitclass,int32, float32,float64

def makeT(cp):
    # cp: [K x 2] control points
    # T: [(K+3) x (K+3)]
    K = cp.shape[0]
    T = np.zeros((K+3, K+3))
    T[:K, 0] = 1
    T[:K, 1:3] = cp
    T[K, 3:] = 1
    T[K+1:, 3:] = cp.T
    R = squareform(pdist(cp, metric='euclidean'))
    R = R * R
    R[R == 0] = 1 # a trick to make R ln(R) 0
    R = R * np.log(R)
    np.fill_diagonal(R, 0)
    T[:K, 3:] = R
    return T

def liftPts(p, cp):
    # p: [N x 2], input points
    # cp: [K x 2], control points
    # pLift: [N x (3+K)], lifted input points
    N, K = p.shape[0], cp.shape[0]
    pLift = np.zeros((N, K+3))
    pLift[:,0] = 1
    pLift[:,1:3] = p
    R = cdist(p, cp, 'euclidean')
    R = R * R
    R[R == 0] = 1
    R = R * np.log(R)
    pLift[:,3:] = R
    return pLift

def test1():

    # source control points
    x, y = np.linspace(-1, 1, 3), np.linspace(-1, 1, 3)
    x, y = np.meshgrid(x, y)
    xs = x.flatten()
    ys = y.flatten()
    cps = np.vstack([xs, ys]).T

    # target control points
    xt = xs + np.random.uniform(-0.3, 0.3, size=xs.size)
    yt = ys + np.random.uniform(-0.9, 0.3, size=ys.size)

    # construct T
    T = makeT(cps)

    # solve cx, cy (coefficients for x and y)
    xtAug = np.concatenate([xt, np.zeros(3)])
    ytAug = np.concatenate([yt, np.zeros(3)])
    cx = nl.solve(T, xtAug) # [K+3]
    cy = nl.solve(T, ytAug)

    # dense grid
    N = 30
    x = np.linspace(-2, 2, N)
    y = np.linspace(-2, 2, N)
    x, y = np.meshgrid(x, y)
    xgs, ygs = x.flatten(), y.flatten()
    gps = np.vstack([xgs, ygs]).T

    # gps = np.array([[-1,-1],[-1.2,0],[0,-1.4]])
    # xgs = gps[:,0]
    # ygs = gps[:,1]

    # transform
    pgLift = liftPts(gps, cps) # [N x (K+3)]
    xgt = np.dot(pgLift, cx.T)
    ygt = np.dot(pgLift, cy.T)

    # display
    plt.xlim(-2.5, 2.5)
    plt.ylim(-2.5, 2.5)
    plt.subplot(1, 2, 1)
    plt.title('Source')
    plt.grid()
    plt.scatter(xs, ys, marker='+', c='r', s=40)
    plt.scatter(xgs, ygs, marker='.', c='r', s=5)
    plt.subplot(1, 2, 2)
    plt.title('Target')
    plt.grid()
    plt.scatter(xt, yt, marker='+', c='b', s=40)
    plt.scatter(xgt, ygt, marker='.', c='b', s=5)
    plt.show()






def print_para_resut(step, timer_end, timer_start):
    print step
    print "in %f sec\n" % (timer_end - timer_start)
def get_binaray_img_boundary(binary_img):
    # Call cv2.findContours'
    import  cv2
    contours, _ = cv2.findContours(binary_img, cv2.RETR_LIST, cv2.cv .CV_CHAIN_APPROX_NONE)
    print contours
    coutour_img = np.zeros((binary_img.shape[0],binary_img.shape[1],3),np.uint8)
    coutour_img[:,:,:] =[255,255,255]
    from time import time
    timer_start = time()
    boundary_pixel =[] #x,y格式
    for j in range(0,binary_img.shape[0]):
        for i in range(0,binary_img.shape[1]):
            is_boundary = False
            if binary_img[j,i] == 0 :
                continue
            if j>0:
                if binary_img[j-1, i] == 0:
                    is_boundary = True
            if j < binary_img.shape[0]-1:
                if binary_img[j+1, i] == 0:
                    is_boundary = True
            if i>0:
                if binary_img[j, i-1] == 0:
                    is_boundary = True
            if i < binary_img.shape[1]-1:
                if binary_img[j, i+1] == 0:
                    is_boundary = True
            if is_boundary:
                boundary_pixel.append([i,j]) # x,y 格式
                binary_img[j,i] = 127
    print  len(boundary_pixel)
    order_boundary = []
    from strand_convert import  get_color_from_rad
    from math import pi
    contours = contours[0]
    for i in range(0,contours.shape[0]):
        x,y = contours[i,0,:]
        print 2*pi*i/contours.shape[0]
        coutour_img[y,x,:] = get_color_from_rad(2*pi*i/contours.shape[0])[::-1]
        if i == 0:
            coutour_img[y, x, :] = [0,0,0]
        if i== contours.shape[0]-1:
            coutour_img[y, x, :] = [127, 127, 127]
    return contours,coutour_img
    start_pixel = boundary_pixel[len(boundary_pixel)-1]
    visited =[]
    while len(boundary_pixel):
        i = start_pixel[0]
        j = start_pixel[1]
        boundary_pixel.remove([i,j])
        visited.append([i,j])
        neibour = []
        find_neibour = False
        if j > 0:
            if  [j - 1, i] in boundary_pixel:
                find_neibour = True
                neibour.append([j - 1, i])
        if j < binary_img.shape[0] - 1 :
            if [j+1, i] in  boundary_pixel:
                find_neibour = True
                neibour.append([j+1, i])
        if i > 0:
            if [j, i-1] in  boundary_pixel:
                find_neibour = True
                neibour.append([j , i-1])
        if i < binary_img.shape[1] - 1:
            if [j, i+1] in  boundary_pixel:
                find_neibour = True
                neibour.append([j, i+1])


    timer_end = time()
    print_para_resut('get_binaray_img_boundary()', timer_end, timer_start)
    return  binary_img

def get_control_point_from_binary_img(binary_path1,binary_path2):
    import cv2
    INPUT1 = cv2.imread(binary_path1,cv2.IMREAD_GRAYSCALE)
    INPUT2 = cv2.imread(binary_path2,cv2.IMREAD_GRAYSCALE)
    height = INPUT1.shape[0]
    contours1,_ = get_binaray_img_boundary(INPUT1)
    contours2,_ = get_binaray_img_boundary(INPUT2)
    contours1 = contours1.reshape((contours1.shape[0],2))
    contours2 = contours2.reshape((contours2.shape[0], 2))
    contours1[:,1] = height-1-contours1[:,1]
    contours2[:, 1] = height - 1 - contours2[:, 1]
    select1 = np.linspace(0,contours1.shape[0]-1,50,True)
    select2 = np.linspace(0, contours2.shape[0]-1, 50, True)
    select1 = select1.astype(np.int)
    select2 = select2.astype(np.int)
    cp1 =contours1[select1,:]
    cp2 =contours2[select2,:]
    return cp1,cp2

def imag_warp(ori_control_point,target_control_point):

    cps = ori_control_point
    # construct T
    T = makeT(cps)
    # target control points
    xt = target_control_point[:,0]
    yt = target_control_point[:,1]
    # solve cx, cy (coefficients for x and y)
    xtAug = np.concatenate([xt, np.zeros(3)])
    ytAug = np.concatenate([yt, np.zeros(3)])
    cx = nl.solve(T, xtAug) # [K+3]
    cy = nl.solve(T, ytAug)
    # dense grid
    N = 100
    x = np.linspace(-2, 500, N)
    y = np.linspace(-2, 500, N)
    x, y = np.meshgrid(x, y)
    xgs, ygs = x.flatten(), y.flatten()
    gps = np.vstack([xgs, ygs]).T
    pgLift = liftPts(gps, cps) # [N x (K+3)]
    xgt = np.dot(pgLift, cx.T)
    ygt = np.dot(pgLift, cy.T)
    # display
    # plt.xlim(-2.5, 500)
    # plt.ylim(-2.5, 500)
    plt.axis("equal")
    plt.subplot(1, 2, 1)
    plt.title('Source')
    plt.grid()

    plt.scatter(ori_control_point[:,0], ori_control_point[:,1], marker='+', c='r', s=40)
    plt.scatter(xgs, ygs, marker='.', c='r', s=5)
    plt.subplot(1, 2, 2)
    plt.title('Target')
    plt.grid()
    plt.scatter(xt, yt, marker='+', c='r', s=40)
    plt.scatter(xgt, ygt, marker='.', c='b', s=5)
    plt.show()


if __name__ == '__main__':

    input_dir ='E:/workspace/dataset/zhengjianzhao_seg_dataset/test_img/'
    output_dir = 'E:/workspace/dataset/zhengjianzhao_seg_dataset/test_img/'
    source_object_name = 'D13010541462008'
    target_object_name='A13010436691206'
    from fitting.util import smooth_seg_image
    #smooth_seg_image(input_dir+source_object_name+'.png',output_dir+source_object_name+'_binary.png')
    # smooth_seg_image(input_dir+target_object_name+'.png',output_dir+target_object_name+'_binary.png')
    ori_control_point = np.array([[],[]])
    x, y = np.linspace(40, 400, 10), np.linspace(40, 400, 10)
    x, y = np.meshgrid(x, y)
    xs = x.flatten()
    ys = y.flatten()
    cps = np.vstack([xs, ys]).T

    # target control points
    xt = xs + np.random.uniform(-30, 30, size=xs.size)
    yt = ys + np.random.uniform(-20, 20, size=ys.size)
    target_cps = np.vstack([xt, yt]).T

    cps,target_cps = get_control_point_from_binary_img(input_dir+source_object_name+'_binary.png',input_dir+target_object_name+'_binary.png')
    imag_warp(cps,target_cps)

    pass

