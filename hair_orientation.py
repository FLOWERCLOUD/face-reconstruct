# -- coding: utf-8 --
import os
import numpy as np
from math import exp,cos,sin,pi,sqrt,fabs,floor,ceil,pi
import  cv2
from time import time
from numba import jit,int64,float64
from functools import wraps
from fitting.util import load_binary_pickle,save_binary_pickle,safe_mkdirs,write_full_obj,convert_img_2_mseh_new


def fn_timer(function):
    @wraps(function)
    def function_timer(*args, **kwargs):
        t0 = time()
        result = function(*args, **kwargs)
        t1 = time()
        print ("Total time running %s: %s seconds" %
               (function.func_name, str(t1-t0))
               )
        return result
    return function_timer

# theta angle
def print_para_resut(step, timer_end, timer_start):
    print step
    print "in %f sec\n" % (timer_end - timer_start)
def k_gabor_kernel_theta(theta,u=1.8,v=2.4):
    theta = theta*pi/180.0  #convert to rad

    sigma_u = 1.8
    sigma_v = 2.4
    lamda = 4.0
    nstds = 1
    sqrt_sigma_x = u
    sqrt_sigma_y = v
    xmax = max(abs(nstds*sqrt_sigma_x*cos(theta)),abs(nstds*sqrt_sigma_y*sin(theta)))
    ymax = max(abs(nstds * sqrt_sigma_x * sin(theta)), abs(nstds * sqrt_sigma_y * cos(theta)))
    xmax = int(xmax)
    ymax = int(ymax)
    half_filter_size = xmax if xmax > ymax else ymax
    filter_size = 2 * half_filter_size + 1
    gaber = np.zeros((filter_size,filter_size,1))
    for y in range(0,filter_size):
        for x in range(0,filter_size):
            x1 = x - half_filter_size
            y1 = y - half_filter_size
            x_theta = x1*cos(theta) + y1*sin(theta)
            y_theta = -x1 * sin(theta) + y1 * cos(theta)
            gaber[y,x] =  exp( -0.5*( x_theta*x_theta/sigma_u/sigma_u  + y_theta*y_theta/sigma_v/sigma_v))*cos(2*pi*y_theta/lamda)
    return gaber
# response of k_theata at pixel (x ,y)
def gaborFilter(image,filter):
    half_filter_size = (max(filter.shape[0], filter.shape[1]) - 1) / 2
    filtered_img = np.zeros((image.shape[0],image.shape[1],image.shape[2]),dtype= np.uint8)
    for i in range(0,image.shape[0]):
        for j in range(0,image.shape[1]):
            filter_value = np.array([0.0,0.0,0.0,0.0])
            for fi  in range(0,filter.shape[0]):
                img_i = i + fi - half_filter_size
                img_i = 0 if img_i < 0 else img_i;
                img_i = image.shape[0] - 1 if img_i >= image.shape[0] else img_i
                for fj in range(0,filter.shape[1]):
                    img_j = j + fj - half_filter_size
                    img_j = 0 if img_j < 0 else img_j
                    img_j = image.shape[1] - 1 if img_j >= image.shape[1] else img_j
                    tmp = image[img_i,img_j,:]*filter[fi,fj]
                    filter_value[:]+=tmp[:]
            filtered_img[i,j,:] =  filter_value
    return filtered_img

#这个ra
def getcolor(ratio):

    #print('should not use this')
    if ratio <0.0 :
        ratio = 0
    if ratio >1.0:
        ratio = 1.0
    return get_undirected_color_from_rad(pi * ratio)
    if ratio < 0.5:
        ratio = ratio/0.5
        return np.array([255, 0, 0]) * (1.0-ratio) + np.array([0, 0, 255]) *ratio
    else:
        ratio = (ratio-0.5)/0.5
        return np.array([0, 0, 255]) * (1.0 - ratio)  + np.array([0, 255, 0]) * ratio

def get_undirected_color_from_rad(rad):
    #颜色排列 rgb
    theta = rad
    theta_range = (np.array([0,pi/3,pi/3*2,pi,pi/3*4,pi/3*5,2*pi]) /2).tolist()
    #color_range = np.array([[0,0,255],[255,0,0],[0,255,0],[0,0,255],[255,0,0],[0,255,0]])
    color_range = np.array([[0, 0, 255],[255, 0, 255],[255, 0, 0],[255, 255, 0],[0, 255, 0],[0, 255, 255] ])
    theta_step = pi/3/2
    if theta <0:
        theta = 0
    if theta> 2*pi:
        theta = 2*pi
    if theta >=pi:
        theta -=pi
    if theta < theta_range[1]:
        i=0
        color = color_range[i]* (1- (theta-theta_range[i])/theta_step) + color_range[i+1] *(theta-theta_range[i])/theta_step
    elif theta < theta_range[2]:
        i=1
        color = color_range[i]* (1- (theta-theta_range[i])/theta_step) + color_range[i+1] *(theta-theta_range[i])/theta_step
    elif theta < theta_range[3]:
        i=2
        color = color_range[i]* (1- (theta-theta_range[i])/theta_step) + color_range[i+1] *(theta-theta_range[i])/theta_step
    elif theta <theta_range[4]:
        i=3
        color = color_range[i]* (1- (theta-theta_range[i])/theta_step) + color_range[i+1] *(theta-theta_range[i])/theta_step
    elif theta < theta_range[5]:
        i=4
        color = color_range[i]* (1- (theta-theta_range[i])/theta_step) + color_range[i+1] *(theta-theta_range[i])/theta_step
    elif theta < theta_range[6]:
        i=5
        color = color_range[i]* (1- (theta-theta_range[i])/theta_step) + color_range[0] *(theta-theta_range[i])/theta_step
    else:
        color = color_range[0]
    return  color


def getcolor_from_rad(rad):

    if rad< 0:
        rad =0
    if rad > 2*pi:
        rad = 2*pi
    return get_undirected_color_from_rad(rad)

def getcolor2(ratio):

    if ratio <0.0 :
        ratio = 0
    if ratio >1.0:
        ratio = 1.0
    if ratio < 0.25:
        rate = (ratio)/0.25
        return  np.array([0,125,125])*rate +np.array([0,125,125])
    elif ratio <0.5:
        rate = (ratio-0.25)/0.25
        return  np.array([0,255,0])*rate +np.array([0,125,0])
    elif ratio < 0.75:
        rate = (ratio-0.5)/0.25
        return  np.array([255,255,0])*rate+np.array([125,125,0])
    else:
        rate = (ratio-0.75)/0.25
        return  np.array([0,0,255])*rate+np.array([0,0,125])

def get_orientation_map( img_path,write_dir ,gabor_num = 32):
    theta_array = np.linspace(0.0,180.0,gabor_num,False)
    gabor_array = []
    timer_start = time()
    for i in range(0,theta_array.size):
        gabor = k_gabor_kernel_theta(theta_array[i])
        gabor_array.append(gabor)
    timer_end = time()
    print_para_resut('build gabor_array', timer_end, timer_start)
    source_image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED )
    for i in range(0,len(gabor_array)):
        timer_start = time()
        result_image = gaborFilter(source_image,gabor_array[i])
        cv2.imwrite(write_dir+str(i)+'.png', result_image[:,:,0])
        timer_end = time()
        print_para_resut('orientaion_ '+ str(i)+'__'+str(theta_array[i]) , timer_end, timer_start)

#@jit()


def get_orientation_map_cv(img_path,write_dir):
    timer_start = time()
    theta_array = np.linspace(0.0,180.0,32,False)
    img = cv2.imread(img_path,cv2.IMREAD_UNCHANGED )

    img_mask = img[:,:,3] > 0

    orientation_img = np.zeros((img.shape[0], img.shape[1], img.shape[2]), dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    filtered_img_array = []
    max_response = np.zeros((img.shape[0],img.shape[1]),dtype = float)
    max_orientaion_idx = np.zeros((img.shape[0],img.shape[1]),dtype = int)
    max_orientaion_idx[:,:] = 33

    max_response = max_response.reshape(max_response.size, 1)
    for i in range(0,len(theta_array)):
        g_kernel = cv2.getGaborKernel((20, 20), 3.0, theta_array[i]*pi/180.0, 4.0, 0.5, 0, ktype=cv2.CV_32F)

        filtered_img = cv2.filter2D(img, cv2.CV_8UC3, g_kernel)
        filtered_img_array.append(filtered_img)
#        cv2.imwrite(write_dir + 'filtered image_' + str(theta_array[i]) + '.png', filtered_img)
#        bool_map = np.zeros((img.shape[0], img.shape[1]), dtype=bool)
        #bool_map[:,:,0] = True if filtered_img[:,:] > max_response[:,:,0] else False
        filtered_img = filtered_img.reshape(filtered_img.size, 1)
        max_orientaion_idx = max_orientaion_idx.reshape(max_orientaion_idx.size, 1)
        max_response = max_response.reshape(max_response.size, 1)
        bool_map = filtered_img[:, :] > max_response[:, :]

        bool_map = bool_map.reshape(bool_map.size, 1)
        max_response = np.array([(x if c else y) for x, y, c in zip(filtered_img[:,0],max_response[:,0],bool_map[:,0])])
        curren_oriten_idx = max_orientaion_idx.copy()
        curren_oriten_idx[:,:] = i
        curren_oriten_idx = curren_oriten_idx.reshape(curren_oriten_idx.size, 1)
        max_orientaion_idx = np.array([(x if c else y) for x, y, c in zip(curren_oriten_idx[:,0],max_orientaion_idx[:,0],bool_map[:,0])])
#        max_response[:,:,0] = filtered_img[:,:,0] if bool_map[:,:,0] else max_response[:,:,0]
#        max_orientaion_idx[:,:,0] = i if bool_map[:,:,0] else max_orientaion_idx[:,:,0]
        # cv2.imshow('image', img)
        # cv2.imshow('filtered image', filtered_img)
#        cv2.imwrite(write_dir + 'filtered image_' + str(theta_array[i]) + '.png', filtered_img)
#        h, w = g_kernel.shape[:2]
#        g_kernel = cv2.resize(g_kernel, (3 * w, 3 * h), interpolation=cv2.INTER_CUBIC)
#        cv2.imwrite(write_dir + 'g_kernel_' + str(i) + '.png', g_kernel)

    max_orientaion_idx = max_orientaion_idx.reshape( img.shape[0],img.shape[1])

    timer_end = time()
    print_para_resut('get f', timer_end, timer_start)

    coeff_map = get_coeff(filtered_img_array, theta_array, max_orientaion_idx)
    print coeff_map.min()
    print coeff_map.max()

    for i in range(0,orientation_img.shape[0]):
        for j in range(0,orientation_img.shape[1]):
            if img_mask[i,j] :
                orientation_img[i,j,0:3] = getcolor(max_orientaion_idx[i,j] /32.0)
                orientation_img[i,j,3] = 255
            else:
                orientation_img[i, j,3] = 0
            if coeff_map[i,j] <20000:
                orientation_img[i, j, 3] = 0

    cv2.imwrite(write_dir + 'orientation_img_refine20000'  + '.png', orientation_img)

@fn_timer
def get_coeff(f_array,theata_array,best_theata_idx,img_mask) :

    coeff_map = np.zeros( (f_array[0].shape[0],f_array[0].shape[1]))
    for i in range(0,best_theata_idx.shape[0]):
        for j in range(0, best_theata_idx.shape[1]):

            if not img_mask[i,j]:
                coeff_map[i, j] == 0.0
                continue
            best_idx =  best_theata_idx[i,j]
            if best_idx >= len(theata_array):
                print i,' ',j,' ',best_idx
                coeff_map[i,j]==0.0
                continue
            distance = 0.0
            for m in range(0,len(f_array)):
                d1 = abs(float(f_array[m][i, j]) - float(f_array[best_idx][i, j]))
                tmp = sqrt( abs(theata_array[m] - theata_array[best_idx]))* d1
                distance +=tmp
            coeff_map[i,j]=distance

    return coeff_map


@fn_timer
def get_response_map(img,theta_array,gabor_para):
    if len(img.shape)>2:
        print img.shape
        if img.shape[2]>2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #Y = 0.299*R+0.587*G +0.114*B
    filtered_img_array = []
    g_kernel_array = []
    max_response = np.zeros((img.shape[0], img.shape[1]), dtype=float)
    max_response[:,:] = -1000.0
    max_orientaion_idx = np.zeros((img.shape[0], img.shape[1]), dtype=int)
    max_orientaion_idx[:, :] = len(theta_array)

    max_response = max_response.reshape(max_response.size, 1)
    for i in range(0, len(theta_array)):
        g_kernel = cv2.getGaborKernel((gabor_para['ksize'], gabor_para['ksize']), gabor_para['sigma'], theta_array[i] * pi / 180.0, gabor_para['lambda'], gabor_para['gamma'], gabor_para['psi'], ktype=cv2.CV_32F)
        g_kernel_array.append(g_kernel)
        filtered_img = cv2.filter2D(img, cv2.CV_8UC3, g_kernel)
        filtered_img_array.append(filtered_img)
        #        cv2.imwrite(write_dir + 'filtered image_' + str(theta_array[i]) + '.png', filtered_img)
        #        bool_map = np.zeros((img.shape[0], img.shape[1]), dtype=bool)
        # bool_map[:,:,0] = True if filtered_img[:,:] > max_response[:,:,0] else False
        filtered_img = filtered_img.reshape(filtered_img.size, 1)
        max_orientaion_idx = max_orientaion_idx.reshape(max_orientaion_idx.size, 1)
        max_response = max_response.reshape(max_response.size, 1)
        bool_map = filtered_img[:, :] > max_response[:, :]

        bool_map = bool_map.reshape(bool_map.size, 1)
        max_response = np.array(
            [(x if c else y) for x, y, c in zip(filtered_img[:, 0], max_response[:, 0], bool_map[:, 0])])
        curren_oriten_idx = max_orientaion_idx.copy()
        curren_oriten_idx[:, :] = i
        curren_oriten_idx = curren_oriten_idx.reshape(curren_oriten_idx.size, 1)
        max_orientaion_idx = np.array(
            [(x if c else y) for x, y, c in zip(curren_oriten_idx[:, 0], max_orientaion_idx[:, 0], bool_map[:, 0])])
        #        max_response[:,:,0] = filtered_img[:,:,0] if bool_map[:,:,0] else max_response[:,:,0]
        #        max_orientaion_idx[:,:,0] = i if bool_map[:,:,0] else max_orientaion_idx[:,:,0]
        # cv2.imshow('image', img)
        # cv2.imshow('filtered image', filtered_img)
        #        cv2.imwrite(write_dir + 'filtered image_' + str(theta_array[i]) + '.png', filtered_img)
        #        h, w = g_kernel.shape[:2]
        #        g_kernel = cv2.resize(g_kernel, (3 * w, 3 * h), interpolation=cv2.INTER_CUBIC)
        #        cv2.imwrite(write_dir + 'g_kernel_' + str(i) + '.png', g_kernel)

    max_orientaion_idx = max_orientaion_idx.reshape(img.shape[0], img.shape[1])
    return filtered_img_array,max_response,max_orientaion_idx,g_kernel_array

#@jit()
def get_Orientationmap_color(max_orientaion_idx,coeff_map,img_mask,theta_array,filter_threshold =20000):
    orientation_img = np.zeros((max_orientaion_idx.shape[0], max_orientaion_idx.shape[1], 4), dtype=np.uint8)
    for i in range(0,orientation_img.shape[0]):
        for j in range(0,orientation_img.shape[1]):
            if img_mask[i,j] :
                color = get_undirected_color_from_rad(theta_array[max_orientaion_idx[i,j]]/180.0*pi)
                orientation_img[i, j, 0:3] = color[::-1] # 转化为 bgr
                #orientation_img[i,j,0:3] = getcolor(max_orientaion_idx[i,j] /32.0)
                orientation_img[i,j,3] = 255
                if max_orientaion_idx[i,j]>=32 or max_orientaion_idx[i,j]<0 :
                    print("max_orientaion_idx %d ." % (max_orientaion_idx[i,j]))
            else:
                orientation_img[i, j,3] = 0  # alpha 设为0
            if coeff_map[i,j] < filter_threshold:
                orientation_img[i, j, 3] = 0 # alpha 设为0
    return orientation_img


@fn_timer
def get_orientation_map_cv_new(img_path,input_seg_img_path,write_dir,scale=0.5):
    theta_array = np.linspace(0.0,180.0,32,False)
    img = cv2.imread(img_path,cv2.IMREAD_UNCHANGED )
    img_seg = cv2.imread(input_seg_img_path,cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    img = cv2.resize(img, (int(scale * w), int(scale * h)), interpolation=cv2.INTER_CUBIC)
    img_mask = np.zeros((img.shape[0], img.shape[1]))
    if img_seg.shape[0] == img.shape[0] and img_seg.shape[1] == img.shape[1]:
        for i in range(0,img_seg.shape[0]):
            for j in range(0,img_seg.shape[1]):
                if img_seg[i,j,0] ==0 and img_seg[i,j,1] == 0 and img_seg[i,j,2] == 255: #bgr 格式 红色
                    img_mask[i,j] = 1
                else:
                    img_mask[i, j] = 0
                    img[i, j, 0:3] = [0, 255, 0] # 把非头发部分颜色值设为绿色
    else:
        print 'seg not exist'
    # if len(img.shape)>2 and img.shape[2]>3:
    #     img_mask = img[:,:,3] > 0
    # else:
    #     img_mask  = np.ones((img.shape[0],img.shape[1]))
    gabor_para ={}
    gabor_para['ksize'] = 20 # size of gabor filter (n, n)
    gabor_para['sigma'] =3.0 #standard deviation of the gaussian function
    gabor_para['theta'] = 0.0
    gabor_para['lambda'] = 4.0
    gabor_para['gamma'] = 0.5
    gabor_para['psi'] = 0

#new para
    gabor_para['ksize'] = 50 # size of gabor filter (n, n)
    gabor_para['sigma'] = 2.0 #1.8 #1.8 standard deviation of the gaussian function
    gabor_para['theta'] = 0.0
    gabor_para['lambda'] = 4.0
    gabor_para['gamma'] = 0.5 #1.8/2.4
    gabor_para['psi'] = 0


    filtered_img_array ,max_response,max_orientaion_idx,g_kernel_array = get_response_map(img,theta_array,gabor_para)
    for i in range(0 ,len(filtered_img_array)):

        cv2.imwrite(write_dir + 'filtered_image_'
                    + 'ksize' + str(gabor_para['ksize']) + 'sigma_' + str(gabor_para['sigma']) + 'theta_'
                    + str(theta_array[i])+ 'lambda_' + str(gabor_para['lambda']) + 'gamma_'
                    + str(gabor_para['gamma']) + 'psi_' + str(gabor_para['psi'])
                    + '.png', filtered_img_array[i])
        g_kernel = g_kernel_array[i]
        h, w = g_kernel.shape[:2]
        g_min = g_kernel.min()
        g_max = g_kernel.max()
        g_range = g_max -g_min
        scale = 255.0/g_range
        g_kernel[:,:] -= g_min
        g_kernel[:, :] *=scale

        g_kernel = cv2.resize(g_kernel, (10 * w, 10 * h), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(write_dir + 'g_kernel_image_'
                    +'ksize'+str(gabor_para['ksize'])+'sigma_'+str(gabor_para['sigma'])+'theta_'+str(theta_array[i])
                    +'lambda_'+str(gabor_para['lambda'])+'gamma_'+str(gabor_para['gamma'])+'psi_'+str(gabor_para['psi'])
                    + '.png', g_kernel)

    max_response= max_response.reshape(img.shape[0],img.shape[1])
    max_orientaion_idx = max_orientaion_idx.reshape(img.shape[0],img.shape[1])
    coeff_map = get_coeff(filtered_img_array, theta_array, max_orientaion_idx,img_mask)
    filter_threshold = 20.0
    t1 = time()
    coeff_map[:,:] = coeff_map[:,:]/coeff_map.max()*255
    cv2.imwrite(write_dir + 'coeff_map_'
                + 'ksize' + str(gabor_para['ksize']) + 'sigma_' + str(gabor_para['sigma'])
                + 'lambda_' + str(gabor_para['lambda']) + 'gamma_'
                + str(gabor_para['gamma']) + 'psi_' + str(gabor_para['psi'])
                +'.png', coeff_map)
    orientation_img = get_Orientationmap_color(max_orientaion_idx,coeff_map,img_mask,theta_array,filter_threshold = filter_threshold)
    print 'get_Orientationmap_color',time()-t1
    cv2.imwrite(write_dir + 'orientation_img_refine_'+str(filter_threshold)
                + 'ksize' + str(gabor_para['ksize']) + 'sigma_' + str(gabor_para['sigma'])
                + 'lambda_' + str(gabor_para['lambda']) + 'gamma_'
                + str(gabor_para['gamma']) + 'psi_' + str(gabor_para['psi'])
                + '.png', orientation_img)
    orientation_image = max_orientaion_idx
    coeff_image =coeff_map[:,:]/255.0
    orientaion_dir = theta_array

    #write image mask
    img_mask1 = np.zeros((img_mask.shape[0],img_mask.shape[1]),dtype=np.uint8)
    orentation_img_mask = np.zeros((img_mask.shape[0], img_mask.shape[1]), dtype=np.uint8)

    for i in range(0,img_mask.shape[0]):
        for j in range(0,img_mask.shape[1]):
            if orientation_image[i,j]<len(theta_array):
                orentation_img_mask[i,j] =255
            if img_mask[i,j]:
                img_mask1[i,j]  =255
            else:
                orientation_image[i,j] = len(theta_array)

    cv2.imwrite(write_dir + 'img_mask_'+ '.png', img_mask1)
    cv2.imwrite(write_dir + 'orentation_img_mask_' + '.png', orentation_img_mask)

    data = {'orientation_image':orientation_image,
            'coeff_image':coeff_image,
            'orientaion_dir':orientaion_dir
            }
    save_binary_pickle(data,write_dir+'/init_strand_'
                       + str(filter_threshold)
                       + 'ksize' + str(gabor_para['ksize']) + 'sigma_' + str(gabor_para['sigma'])
                       + 'lambda_' + str(gabor_para['lambda']) + 'gamma_'
                       + str(gabor_para['gamma']) + 'psi_' + str(gabor_para['psi'])
                       + '.pkl'
                       )
    return
    filtered_img_array, max_response, max_orientaion_idx = get_response_map(coeff_map, theta_array)
    max_response= max_response.reshape(img.shape[0],img.shape[1])
    max_orientaion_idx = max_orientaion_idx.reshape(img.shape[0],img.shape[1])
    coeff_map = get_coeff(filtered_img_array, theta_array, max_orientaion_idx)
    filter_threshold = 20.0
    t1 = time()
    coeff_map[:,:] = coeff_map[:,:]/coeff_map.max()*255
    cv2.imwrite(write_dir + 'coeff_map_iter1' +'.png', coeff_map)
    orientation_img = get_Orientationmap_color(max_orientaion_idx,coeff_map,img_mask,filter_threshold = filter_threshold)
    print 'get_Orientationmap_color',time()-t1
    cv2.imwrite(write_dir + 'orientation_img_refine_iter1'+str(filter_threshold)  + '.png', orientation_img)

def strand_trace(pickel_input_path ,output_file_dir):
    data = load_binary_pickle(pickel_input_path)
    orientation_image = data['orientation_image']
    coeff_image = data['coeff_image']
    orientaion_dir = data['orientaion_dir']
    #conver to 左下角坐标
    orientation_image[:,:] = orientation_image[::-1,:] #实现上下翻转
    coeff_image[:,:] = coeff_image[::-1,:]
    #要初始化一下这里
    Strand.max_strand_idx = -1
    Strand.begin_strand_idx = 10 # 头发序号计数开始值
    strand_tracer = Strand_Trace(orientation_image, coeff_image, orientaion_dir)
    strand_tracer.init_seed_pixel(2000)
#    for i in range(10,2000,100):
#        strand_tracer.write_seed_to_image(output_file_dir,i)

    strand_tracer.start_trace()
    #strand_tracer.write_coeff_image(output_file_dir)
    #strand_tracer.write_varid_image(output_file_dir)
    #strand_tracer.write_orientation_image(output_file_dir,coeff_threshold = 0.15)
    #输出前10根最长头发
    if 0:
        safe_mkdirs(output_file_dir+'/max_len_strand/')
        for i in range(0,10):
            strand_tracer.write_strand_to_image(output_file_dir+'/max_len_strand/', idx= i,stran_len_threshold=50)
        #输出长度超过阈值的头发
        safe_mkdirs(output_file_dir + '/threshold_strand/')
        strand_tracer.write_strand_to_image(output_file_dir + '/threshold_strand/', idx=-1, stran_len_threshold=1)
        strand_tracer.write_strand_to_image(output_file_dir + '/threshold_strand/', idx=-1, stran_len_threshold=3)
        strand_tracer.write_strand_to_image(output_file_dir + '/threshold_strand/', idx=-1, stran_len_threshold=10)
        strand_tracer.write_strand_to_image(output_file_dir + '/threshold_strand/', idx=-1, stran_len_threshold=30)
        strand_tracer.write_strand_to_image(output_file_dir + '/threshold_strand/', idx=-1, stran_len_threshold=50)



        # 输出 头发及颜色
        safe_mkdirs(output_file_dir + '/strand_with_color/')
        strand_tracer.write_strand_to_image(output_file_dir + '/strand_with_color/', idx=-1, stran_len_threshold=1,use_theta_color=True,show_seed = False)
        strand_tracer.write_strand_to_image(output_file_dir + '/strand_with_color/', idx=-1, stran_len_threshold=3,use_theta_color=True,show_seed = False)
        strand_tracer.write_strand_to_image(output_file_dir + '/strand_with_color/', idx=-1, stran_len_threshold=10,use_theta_color=True,show_seed = False)
        strand_tracer.write_strand_to_image(output_file_dir + '/strand_with_color/', idx=-1, stran_len_threshold=30,use_theta_color=True,show_seed = False)
        strand_tracer.write_strand_to_image(output_file_dir + '/strand_with_color/', idx=-1, stran_len_threshold=50,use_theta_color=True,show_seed = False)
        #for i in range(0,100):
        #    strand_tracer.write_strand_to_image(output_file_dir,idx = i)
    return strand_tracer.strands

'''
orientation_image ： M*N*1 ,整形，表示的方向的序号
coeff_image: M*N*1 ,浮点序，该像素点 属于该方向序号的概率
orientaion_dir : k*1 序号对应的角度
'''

class Pixel:
    def __init__(self,y,x,coeff,ori_idx,theta,isvalid = True):
        self.y = y # 左下角为原点
        self.x = x
        self.coeff = coeff
        self.ori_idx = ori_idx
        self.theta = theta
        self.belong_strand_idx = -1
        self.isvalid = isvalid # 用于确认这个pixel是否是可用的，否则可为背景pixel
        self.isSeed = False
    def set_strand_idx(self,idx):
        self.belong_strand_idx = idx
    def __cmp__(self, pe):
        if __lt__(pe) == True:
            return -1
        elif  __eq__(pe) == True:
            return 0
        else:
            return 1
    def __lt__(self, pe):
        if self.coeff < pe.coeff:
            return  True
        else :
            return False
    def __gt__(self, pe):
        if self.coeff > pe.coeff:
            return True
        else:
            return  False
    def __eq__(self, pe):
        if self.y == pe.y and self.x == pe.x:
            return True
        else:
            return False

    def __hash__(self):
        return hash(str(self.y) + " " + str(self.x))
'''
要求传入的数据坐标都转化为以左下角为原点
'''

class Strand:
    max_strand_idx = -1
    begin_strand_idx = 10 # 头发序号计数开始值
    def __init__(self,strand_idx = -1):
        self.strand =[]
        self.strand_idx = strand_idx
        self.status = 1 # 1 certain,0 uncertain ,2 specail occlude flag
        self.health_point = 5
        self.uncertain_pixel =[] #若health_point 将至0，应从strand中清除这些pixel
        self.seed_pixel =[]
    def add_pixel(self,pix):
        self.strand.append(pix)
    def remove_pixel_array(self,pix_array):
        for pixel in pix_array:
            if pixel in self.uncertain_pixel:
                self.uncertain_pixel.remove(pixel)

    def strand_len(self):
        return len(self.strand)
    def get_cor_y_from_x(self,x):
        x = int(x)
        y = []
        for i in range(0,len(self.strand)):
            pixel = self.strand[i]
            if pixel.x  == x:
                y.append(pixel.y)
        return  y
    def get_cor_x_from_y(self,y):
        y = int(y)
        x = []
        for i in range(0,len(self.strand)):
            pixel = self.strand[i]
            if pixel.y  == y:
                x.append(pixel.x)
        return  x

#参考 https://zhuanlan.zhihu.com/p/30553006
def get_move_line(x0,y0,x1,y1):
    x0 = int(round(x0))
    y0 = int(round(y0))
    x1 = int(round(x1))
    y1 = int(round(y1))
    dx = abs(x1 - x0)
    sx =  1 if x0 < x1  else -1
    dy = abs(y1 - y0)
    sy =  1 if y0 < y1  else -1
    '''
    c 语言中浮点转整数是直接取整数部分,  -1/2 = 0  -(1/2) = 0 (-1)/2 =0
    python int(-0.5) = 0 int(-1/2) = -1 ,int(-(1/2)) = 0 int( (-1)/2) = -1
    '''

    err = int(float(dx)/2) if dx > dy else  int(-float(dy)/2)
    line_pixel = []
    while True:
        line_pixel.append([y0, x0])
        if x0 == x1 and y0 == y1:
            break
        e2 = err
        if (e2 > -dx):
            err -= dy
            x0 += sx
        if (e2 < dy):
            err += dx
            y0 += sy

    return line_pixel
class Strand_Trace:
    def __init__(self,orientation_image,coeff_image,orientaion_dir):

        self.w_high = 0.3
        self.w_low = 0.1
        self.theta_max = 30
        self.orientation_image = orientation_image
        self.coeff_image =coeff_image
        self.orientaion_dir = orientaion_dir
        self.pixels = []
        for i in xrange(0,self.orientation_image.shape[0]):
            row_pixels = []
            for j in xrange(0, self.orientation_image.shape[1]):
                if self.orientation_image[i, j] > len(self.orientaion_dir)-1:
                    pixel = Pixel(i, j, self.coeff_image[i, j], self.orientation_image[i, j],0,False)
                else:
                    pixel = Pixel(i, j, self.coeff_image[i, j], self.orientation_image[i, j],
                                  self.orientaion_dir[self.orientation_image[i, j]],True)
                row_pixels.append(pixel)
            self.pixels.append(row_pixels)
        self.pixels = np.array(self.pixels)
        self.seed_pixel = np.array([])
        # 0 代表没有访问，1 代表已访问
        self.access_mask = np.zeros((orientation_image.shape[0],orientation_image.shape[1]),dtype= int)
        self.strands =[]

    '''
    根据coeff_image 获得数值高的pixel 作为种子点
    '''
    def write_coeff_image(self,file_dir):
        coeff_image = self.coeff_image[::-1,:]*255
        cv2.imwrite(file_dir + '/coeff'  + '.png', coeff_image)
    def write_varid_image(self,file_dir):
        valid_image = np.zeros((self.pixels.shape[0], self.pixels.shape[1]), dtype=np.uint8)
        for i in xrange(0,self.pixels.shape[0]):
            for j in xrange(0,self.pixels.shape[1]):
                pixel = self.pixels[i,j]
                if pixel.isvalid:
                    valid_image[self.pixels.shape[0]-1-i,j] = 255
        cv2.imwrite(file_dir + '/valid_image' + '.png', valid_image)
    def write_orientation_image(self,file_dir,coeff_threshold= 0.07):
        orientaion_image = np.zeros((self.pixels.shape[0], self.pixels.shape[1],3), dtype=np.uint8)
        for i in xrange(0,self.pixels.shape[0]):
            for j in xrange(0,self.pixels.shape[1]):
                pixel = self.pixels[i,j]
                if pixel.coeff < coeff_threshold :
                    orientaion_image[self.pixels.shape[0] - 1 - i, j] = [255,255,255]
                    continue
                #color = getcolor(pixel.theta/180.0)
                color = get_undirected_color_from_rad(pixel/180.0*pi)
                if pixel.isvalid:
                    orientaion_image[self.pixels.shape[0]-1-i,j] = color
        cv2.imwrite(file_dir + '/orientaion_image_' + 'coeff_threshold_'+str(coeff_threshold)+'.png', orientaion_image)


    def write_seed_to_image(self,file_dir,num_seed = 10):
        seed_image = np.zeros((self.coeff_image.shape[0], self.coeff_image.shape[1]), dtype=np.uint8)
        count = 0
        for seed_pixel in self.seed_pix:
            seed_image[self.coeff_image.shape[0] - 1 - seed_pixel.y, seed_pixel.x] = 255
            count += 1
            if count >= num_seed:
                break
        cv2.imwrite(file_dir + '/seed_' + str(num_seed) + '.png', seed_image)
    def write_strand_to_image(self,file_dir,idx = 0,stran_len_threshold = 5,use_theta_color = False,show_seed = True):

        if idx > -1 and idx < len(self.strands):

            strand_image = np.zeros( (self.coeff_image.shape[0],self.coeff_image.shape[1],3),dtype= np.uint8)
            strand = self.strands[idx]
            if not len(strand.strand) >= stran_len_threshold :
                print idx,len(strand.strand)
                return
            for pixel in strand.strand:
                if use_theta_color:
                    theta_color = get_undirected_color_from_rad(pixel.theta / 180.0 * pi)
                    strand_image[self.coeff_image.shape[0] - 1 - pixel.y, pixel.x, :] = theta_color[::-1]
                else:
                    strand_image[self.coeff_image.shape[0] -1 -pixel.y,pixel.x] = [255,255,255] #convert to 左上角坐标
            if show_seed:
                for seed_pixel in strand.seed_pixel:
                    strand_image[self.coeff_image.shape[0] - 1 - seed_pixel.y, seed_pixel.x] = [127,127,127]
            cv2.imwrite(file_dir + '/strand_' + str(idx) +'strandlen'+str(len(strand.strand))+ '_thres_' + str(stran_len_threshold) + '_version2_' + '.png',
                        strand_image)
        elif idx == -1:
            strand_image = np.zeros((self.coeff_image.shape[0], self.coeff_image.shape[1],3), dtype=np.uint8)
            all_strand_num =  len(self.strands)
            print 'all_strand_num',all_strand_num
            count = 0
            for strand in self.strands:
                if not len(strand.strand) >= stran_len_threshold:
                    continue
                color = getcolor( float(count)/10)
                count += 1
                count %=10
                for pixel in strand.strand:
                    if use_theta_color:
                        theta_color = get_undirected_color_from_rad(pixel.theta/180.0*pi)
                        strand_image[self.coeff_image.shape[0] - 1 - pixel.y, pixel.x, :] = theta_color[::-1]
                    else:
                        strand_image[self.coeff_image.shape[0] - 1 - pixel.y, pixel.x,:] = color  # convert to 左上角坐标
                if show_seed:
                    for seed_pixel in strand.seed_pixel:
                        strand_image[self.coeff_image.shape[0] - 1 - seed_pixel.y, seed_pixel.x] = [255,255,255]

            cv2.imwrite(file_dir+'/strand_'+str(idx)+'_thres_'+str(stran_len_threshold)+'_version2_'+'.png',strand_image)


    def init_seed_pixel(self,num = 10):
        self.sorted_pixel =[]
        for i in xrange(0,self.coeff_image.shape[0]):
            for j in xrange(0,self.coeff_image.shape[1]):
                #pixel = Pixel(i,j,self.coeff_image[i,j],self.orientaion_dir[i,j])
                pixel = self.pixels[i,j]
                self.sorted_pixel.append(pixel)
        self.sorted_pixel.sort(reverse=True)
        self.seed_pix = self.sorted_pixel[:]
        # use to test
#        self.seed_pix =[]
#        self.seed_pix .append(self.pixels[564, 415]) #y,x
#        self.seed_pix.append(self.pixels[290, 191])  #291, 191

        seed_pix_check =[]
        for i in range(0,len(self.seed_pix)):
            if self.check_pixel_if_a_seed(self.seed_pix[i]):
                seed_pix_check.append(self.seed_pix[i])
        if len(seed_pix_check) >= num:
            self.seed_pix = seed_pix_check[0:num]
        else:
            self.seed_pix = seed_pix_check[:]
#        self.seed_pix =[]
#        self.seed_pix.append(self.pixels[343, 136]) #(self.pixels[317, 148])  # y,x

    def start_trace(self):
        for i in range(0,len(self.seed_pix)):
            pixel = self.seed_pix[i]
            if pixel.belong_strand_idx >= Strand.begin_strand_idx:
                print pixel.belong_strand_idx, Strand.begin_strand_idx,len(self.strands)
                strand = self.strands[pixel.belong_strand_idx-Strand.begin_strand_idx]
            else:
                strand = Strand()
                if Strand.max_strand_idx <0:
                    Strand.max_strand_idx =  Strand.begin_strand_idx
                else:
                    Strand.max_strand_idx +=1
                strand.strand_idx = Strand.max_strand_idx
                self.strands.append(strand)
            if pixel.y == 343 and pixel.x == 135:
                pass
            strand.status = 1  #初始状态
            strand.health_point = 5
            self.seed_pix[i].isSeed = True
            self.start_trace_single(strand ,self.seed_pix[i],self.seed_pix[i].theta)
            strand.status = 1
            strand.health_point = 5
            self.seed_pix[i].isSeed = True
            strand.seed_pixel.append(self.seed_pix[i])
            self.start_trace_single(strand, self.seed_pix[i],self.seed_pix[i].theta+180.0)
        # 按照 strand 长度从大到小排序
        self.strands.sort(key = lambda x:len(x.strand), reverse=True)

    def get_fill_pixel(self,start_pixel, best_pixel):
        fill_pixels = []
        delta_y = abs(best_pixel.y - start_pixel.y)
        sign_target_g_start_y = True if best_pixel.y > start_pixel.y else False
        delta_x = abs(best_pixel.x - start_pixel.x)
        sign_target_g_start_x = True if best_pixel.x > start_pixel.x else False
        if delta_y == 1 and delta_x == 1:
            pixel1 = self.pixels[start_pixel.y, best_pixel.x]
            pixel2 = self.pixels[best_pixel.y, start_pixel.x]
            fill_pixels.append(pixel1)
            fill_pixels.append(pixel2)
        elif delta_x <= 1 and delta_y > 1:
            for i in xrange(1, delta_y):
                if sign_target_g_start_y :
                    pixel1 = self.pixels[start_pixel.y + i, start_pixel.x]
                else:
                    pixel1 = self.pixels[start_pixel.y - i, start_pixel.x]
                fill_pixels.append(pixel1)
        elif delta_x > 1 and delta_y <= 1:
            for i in xrange(1, delta_x):
                if sign_target_g_start_x:
                    pixel1 = self.pixels[start_pixel.y, start_pixel.x + i]
                else:
                    pixel1 = self.pixels[start_pixel.y, start_pixel.x - i]
                fill_pixels.append(pixel1)
        elif delta_x == 2 and delta_y == 2:
            if sign_target_g_start_x:
                t_y = start_pixel.y +1
            else:
                t_y = start_pixel.y - 1
            if sign_target_g_start_y :
                t_x = start_pixel.y + 1
            else:
                t_x = start_pixel.y - 1
            pixel1 = self.pixels[t_y, t_x]
            fill_pixels.append(pixel1)
        elif delta_x == 0 or delta_y == 0:
            pass
        else:
            print  'dist exceed 2', delta_x, delta_y
        return fill_pixels
    def start_trace_single(self, strand, start_pixel,theta):
    #        if self.access_mask[start_pixel.y,start_pixel.x] !=0:
#            return
        if not start_pixel.isvalid:
            return
        if strand.status != 1 :
            strand.health_point -=1
        if strand.health_point < 1:
            print 'strand.health_point ==0'
            if len(strand.uncertain_pixel) > 0:
                strand.uncertain_pixel = []
            return
        if start_pixel.belong_strand_idx <0 and start_pixel not in strand.uncertain_pixel:
            start_pixel.set_strand_idx(strand.strand_idx)
            strand.add_pixel(start_pixel)
        elif start_pixel.isSeed or start_pixel  in strand.uncertain_pixel: #如果它是seed ，那让它继续
            pass
        else:# 说明已经有strand 包含这个pixel
            return
        next_x ,next_y = self.get_move_dir_pixel( start_pixel.x ,start_pixel.y,theta,2)
        move_pixels = get_move_line(start_pixel.x, start_pixel.y, next_x, next_y)
        move_pixels.remove([start_pixel.y,start_pixel.x])
        has_pixel = False
        smallest_angle = 100.0

        equal_theta = theta
        if equal_theta >=180.0:
            equal_theta -= 180.0
        for y,x in move_pixels:
            next_pixel = self.pixels[y, x]
            if not next_pixel.isvalid:
                continue
            if abs(next_pixel.y - start_pixel.y)<3 and \
                            abs(next_pixel.x - start_pixel.x) <3: #and \
                            #abs(next_pixel.theta - equal_theta) < 10:
                if abs(next_pixel.theta - equal_theta) < smallest_angle:
                    smallest_angle = abs(next_pixel.theta - equal_theta)
                    best_pixel = next_pixel
                    has_pixel = True

        if has_pixel:

            def fill_path():
                fill_pixels = self.get_fill_pixel(start_pixel,best_pixel)
                fill_pixel(fill_pixels)

            def fill_pixel(fill_pixels):
                for pixel2fill in fill_pixels:
                    if pixel2fill.belong_strand_idx < 0:
                        pixel2fill.set_strand_idx(strand.strand_idx)
                        strand.add_pixel(pixel2fill)

            if best_pixel.coeff < self.w_low:
                strand.status = 0 # changed to uncertain
                fill_pixels = self.get_fill_pixel(start_pixel, best_pixel)
                strand.uncertain_pixel = strand.uncertain_pixel+ fill_pixels + [start_pixel,best_pixel] #记录不确定的点
                print 'uncertain'

            if smallest_angle >= self.theta_max:
                strand.status = 2
                fill_pixels = self.get_fill_pixel(start_pixel, best_pixel)
                strand.uncertain_pixel = strand.uncertain_pixel+ fill_pixels + [start_pixel,best_pixel]  # 记录不确定的点
                print 'occlude'

            if best_pixel.coeff >  self.w_low and smallest_angle < self.theta_max:
                strand.status = 1
                strand.health_point = 5
                if len(strand.uncertain_pixel)>0:
                    fill_pixel(strand.uncertain_pixel)
                    print 'fill ',len(strand.uncertain_pixel)
                    strand.uncertain_pixel = []
                # 对置信度比较高的位置的周边进行填补
                fill_path()

            if theta >=180.0:
                self.start_trace_single(strand,best_pixel,best_pixel.theta+180.0)
            else:
                self.start_trace_single(strand, best_pixel, best_pixel.theta)

    def check_pixel_if_a_seed(self,pixel,epslion = 0.2):
        if not pixel.isvalid :
            return  False
        w_p = pixel.coeff
        if  w_p <= self.w_high:
            return  False
        dir_idx = self.orientation_image[pixel.y,pixel.x]
        move_dir = self.orientaion_dir[dir_idx]
        normal_dir = move_dir+90.0
        if normal_dir>180.0:
            normal_dir -= 180.0
        oppose_normal_dir = normal_dir +180.0
        x_new, y_new = self.get_move_dir_pixel(pixel.x,pixel.y,normal_dir,step = 2.0)
        move_line = get_move_line(pixel.x,pixel.y, x_new, y_new)
        move_line.remove([pixel.y,pixel.x])
        w_normal =  [self.coeff_image[y,x] for y,x in move_line]
        x_new, y_new = self.get_move_dir_pixel(pixel.x,pixel.y,oppose_normal_dir,step = 2.0)
        move_line = get_move_line(pixel.x,pixel.y, x_new, y_new)
        move_line.remove([pixel.y, pixel.x])
        w_oppo_normal = [self.coeff_image[y, x] for y, x in move_line]
        w_nomal_all=w_normal[:]+w_oppo_normal[:]
        ratio = (w_p - max(w_nomal_all)) /w_p
        if ratio> epslion and w_p > self.w_high:
            return  True
        else:
            return  False

    # 左下角为原点

    '''
    theta 角度是 y轴 顺时针 0—360度
    x0,y0是起点
        x_new  = x+tan(theta)*delta_y
        y_new = y + delta_y
        delta_y = step*cos(theta)
        故 x_new  = x+step*sin(theta)

    '''

    def get_move_dir_pixel(self,x0,y0,theta,step =1.0):
        delta_y = step * cos(theta*pi/180.0)
        y_new = y0+delta_y
        x_new = x0+step*sin(theta*pi/180.0)
        if x_new < 0:
            x_new = 0
        if x_new> self.coeff_image.shape[1]-1:
            x_new = self.coeff_image.shape[1]-1
        if y_new < 0:
            y_new = 0
        if y_new> self.coeff_image.shape[0]-1:
            y_new = self.coeff_image.shape[0]-1

#        y_new = ceil(y_new)
#        x_new = ceil(x_new)
        return  x_new,y_new










@jit()
def testsum():
    s= 0.0
    #s = 0
    for k in range(1,1000000000): #000000
        s/3.0
        s+=1.0*k
        #s += k
    return s

def test1():
    img  = np.zeros((10000,10000,1))
    for i in xrange(0,img.shape[0]):
        for j in xrange(0,img.shape[1]):
            pass
            #if (i+j )% 10 == 0:
             #   img[i, j] *=2.0
@jit()
def test2():
    img  = np.zeros((10000,10000,1))
    for i in xrange(0,img.shape[0]):
        for j in xrange(0,img.shape[1]):
            pass
            #if (i+j )% 10 == 0:
                #img[i, j] *=2.0
def test3():
    img  = np.zeros((1000,1000,1))
    img[:,:]*=2

def test4():
     img = np.zeros((1000*1000,1))
     #img = img.reshape(img.size,1)
     [x for x in img ]

'''
要求传入的数据坐标都转化为以左下角为原点
'''
class Stroke:
    # [ [xy],[x,y]] 2列
    def __init__(self,stroke,img_width,img_height,strand_idx = -1,revert_y_coordinate = True):
        if len(stroke) >0:
            #去除可能相同的点
            add_vertex =[]
            for v in stroke:
                #v = stroke[i]
                if v in add_vertex:
                    continue
                else:
                    add_vertex.append(v)
            self.stroke =np.array(add_vertex)
            if revert_y_coordinate:
                for i in range(0,self.stroke.shape[0]):
                    self.stroke[i,1] = img_height -1 -self.stroke[i,1]
            self.stroke_idx = strand_idx
            self.intrinsic_direction =[] # 角度rad ,0 -2*pi
            self.intrinsic_2d_direction =[]
            self.dir = 0 # 1 代表目前方向和stroke起点到终点方向 一致， -1代表相反 ,0 代表应重新初始化
            self.resolve_dir()
            self.len =  self.strand_len()
        else:
            self.stroke=np.array([])
            self.stroke_idx = strand_idx
            self.intrinsic_direction =np.array([]) # 角度rad ,0 -2*pi
            self.intrinsic_2d_direction =np.array([])
            self.dir = 0
            self.len = 0
    def revert_dir(self):
        self.dir = -1
        self.resolve_dir()
    def resolve_dir(self):
        if self.stroke.size <3:
            print 'only one vertex 只有一个点'
            print 'self.stroke.shape ', self.stroke.shape, 'self.stroke.size ', self.stroke.size
            return
        if self.dir ==-1:
            self.stroke = self.stroke[::-1,:]
        if self.dir == 1:
            return
        if self.dir == 0:
            pass
        self.intrinsic_direction = np.zeros(self.stroke.shape[0])
        self.intrinsic_2d_direction = np.zeros((self.stroke.shape[0],2))
        for i in range(1,self.stroke.shape[0]-1):
            end_point =  self.stroke[i +1, :]
            start_point = self.stroke[i-1, : ]
            dir_2d = end_point-start_point

            self.intrinsic_2d_direction[i,:] = dir_2d
            rad = strand_convert.get_rad_from2d_dir(dir_2d)
            self.intrinsic_direction[i] = rad
            pass
        #处理起点

        end_point = self.stroke[1, :]
        start_point = self.stroke[0, :]
        dir_2d = end_point - start_point
        if np.linalg.norm(dir_2d) < 0.00001:
            self.intrinsic_2d_direction[0, :] = self.intrinsic_2d_direction[1, :]
            self.intrinsic_direction[0] = self.intrinsic_direction[1]
        else:
            self.intrinsic_2d_direction[0, :] = dir_2d
            rad = strand_convert.get_rad_from2d_dir(dir_2d)
            self.intrinsic_direction[0] = rad
        #处理终点
        end_point = self.stroke[self.stroke.shape[0]-1, :]
        start_point = self.stroke[self.stroke.shape[0]-1-1, :]
        dir_2d = end_point - start_point
        if np.linalg.norm(dir_2d) < 0.00001:
            self.intrinsic_2d_direction[self.stroke.shape[0]-1, :] = self.intrinsic_2d_direction[self.stroke.shape[0]-2, :]
            self.intrinsic_direction[self.stroke.shape[0]-1] = self.intrinsic_direction[self.stroke.shape[0]-2]
        else:
            self.intrinsic_2d_direction[self.stroke.shape[0]-1, :] = dir_2d
            rad = strand_convert.get_rad_from2d_dir(dir_2d)
            self.intrinsic_direction[self.stroke.shape[0]-1] = rad
        self.dir = 1

    def add_vertex(self,v):
        if self.stroke.shape[0] == 0:
            self.stroke = np.array(v)
        else:
            stroke_list = self.stroke.tolist()
            if v in stroke_list:
                print 'stroke_list ',stroke_list
                print 'aready exist',v
                return
            self.stroke = np.vstack((self.stroke,v))
        self.len = self.strand_len()

    #使用线段长来计算
    def strand_len(self):
        #return self.stroke.shape[0]'
        if self.stroke.size <1:
            return 0
        len1 = 0
        for i in range(0,self.stroke.shape[0]-1):
            len1 += np.linalg.norm(self.stroke[i]- self.stroke[i+1] )

        return  len1
    def lapcain_smooth(self,iter_num =1):
        if self.stroke.size <6:
            return
        if self.stroke.shape[0] <3:
            return
        for c in range(0,iter_num):
            for i in range(1,self.stroke.shape[0]-1):
                self.stroke[i,:] = 0.25* self.stroke[i-1,:]+\
                                   0.5*self.stroke[i,:]+\
                                   0.25* self.stroke[i+1,:]
        self.dir = 0
        self.resolve_dir()
'''
输入的strand 只是pixel 的集合，我们需要把它转换为有向的线段
'''
class StrandBBoxf:
    def __init__(self,strand):
        self.min = np.array([100000.0, 100000.0])
        self.max = np.array([-100000.0, -100000.0])
        x_coord = []
        y_coord = []
        for i in range(0,len(strand.strand)):
            pixel = strand.strand[i]
            x_coord.append(pixel.x)
            y_coord.append(pixel.y)
        self.min[0] = min(x_coord)
        self.min[1] = min(y_coord)
        self.max[0] = max(x_coord)
        self.max[1] = max(y_coord)

def convertStrand2Stroke(strand_set,img_width,img_height):
    stroke_set =[]
    for i in range(0,len(strand_set)):
        new_stroke = single_strand2stroke(strand_set[i], img_width, img_height)

        stroke_set.append(new_stroke)

    return stroke_set

def single_strand2stroke(single_strand,img_width,img_height):
    #构建2d包围盒
    bbox_2d = StrandBBoxf(single_strand)
    left_coord = bbox_2d.min
    right_coord = bbox_2d.max
    width = abs(left_coord[0]-right_coord[0])
    height = abs(left_coord[1]-right_coord[1])
    new_stroke = Stroke([],img_width,img_height,strand_idx = -1,revert_y_coordinate = False)
    if height>width: #说明头发是上下走向
        #假设起点位于上方
        y_start = int(bbox_2d.max[1])
        y_end = int(bbox_2d.min[1])
        if y_start > img_height-1:
            y_start = img_height-1
        if y_end <0:
            y_end = 0
        for y in range(y_start,y_end-1,-1):
            x_corr =single_strand.get_cor_x_from_y(y)
            if len(x_corr) == 0:
                #print 'error, get_cor_x_from_y zero',y
                continue
            x = x_corr[0] #选取其中一个
            new_stroke.add_vertex([float(x),float(y)])
            pass
    else: #说明头发是左右走向
        #假设起点位于左边
        x_start = int(bbox_2d.min[1])
        x_end = int(bbox_2d.max[1])
        if x_end > img_width-1:
            x_end = img_width-1
        if x_start <0:
            x_start = 0
        for x in range(x_start,x_end+1):
            y_corr =single_strand.get_cor_y_from_x(x)
            if len(y_corr) == 0:
                #print 'error, get_cor_y_from_x zero',x
                continue
            y = y_corr[0] #选取其中一个
            new_stroke.add_vertex([float(x),float(y)])
    new_stroke.resolve_dir()

    return new_stroke


#这样子计算不大合理
def caculate_stroke_distance(stroke1,stroke2):
    min_dist = 100000.0
    #1. 使用简单的最近点距离
    for i in range(0,stroke1.stroke.shape[0]):
        for j in range(0, stroke2.stroke.shape[0]):
            vertex_dis = np.linalg.norm(stroke1.stroke[i,:] - stroke2.stroke[j,:])
            if vertex_dis < min_dist:
                min_dist = vertex_dis
    return min_dist
''''
stroke1: which dir is unsolve ,can revert
stroke2: resolve stroke
score ,shoud_revert_stroke
'''
def caculate_stroke_similarity(stroke1,stroke2):
    #按理来说，只需 neigh.fit(source_stoke.stroke) 一次
    from sklearn.neighbors import NearestNeighbors
    source_stoke = stroke1
    target_stroke = stroke2
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(source_stoke.stroke)
    distances, indices = neigh.kneighbors(target_stroke.stroke, return_distance=True)
    #筛选出合适的对应 ，考虑的是点到点方向与stroke的角度
    #我这里假设stroke 存的角度是0~2*pi, 这个角度不一定是对的，但假设已经根据stroke的起点和终点计算了出来
    #1 代表角度使用起点和终点的计算结果
    naive_score = 0
    naive_reverse_score = 0

    for target_idx in range(0,target_stroke.stroke.shape[0]):
        target_point_rad = target_stroke.intrinsic_direction[target_idx]
        source_point_rad = source_stoke.intrinsic_direction[indices[target_idx]]
        # if source_point_rad > pi :
        #     source_point_rad -= pi
        # if target_point_rad > pi:
        if source_point_rad < 0 or source_point_rad > 2 * pi or target_point_rad < 0 or target_point_rad > 2 * pi:
            print 'source_point_rad,target_point_rad rad wrong ', source_point_rad, target_point_rad
        naive_score += cos(abs(source_point_rad-target_point_rad))
        reverse_source_point_rad = source_point_rad+pi
        if reverse_source_point_rad > 2*pi:
            reverse_source_point_rad = reverse_source_point_rad -2*pi
        if reverse_source_point_rad <0 or reverse_source_point_rad >2*pi or target_point_rad <0 or target_point_rad >2*pi:
            print 'reverse_source_point_rad ,target_point_rad rad wrong ',reverse_source_point_rad,target_point_rad
        naive_reverse_score += cos(abs(reverse_source_point_rad - target_point_rad))
    if naive_score >= naive_reverse_score:
        return naive_score,False
    if naive_score < naive_reverse_score:
        return naive_score, True


#纠正amb stroke 的方向
def correct_stroke_dir(ambi_stroke,resolve_stroke,revert_stroke):
    if revert_stroke:
        ambi_stroke.revert_dir()
    return ambi_stroke

#hair_array  是list包list
def resolve_strand_direction(strands_set,strock_dir_img,hair_array):
    strock_dir_img = strock_dir_img[::-1,:] #转化为左下角原定
    img_width  = strock_dir_img.shape[1]
    img_height =  strock_dir_img.shape[0]
    strokes = []
    for i in range(0,len(hair_array)):
        strand_array = hair_array[i]
        stroke = Stroke(strand_array,img_width,img_height,strand_idx = i,revert_y_coordinate=True)
        strokes.append(stroke)
    # 按照 stroke 长度从大到小排序
    strokes.sort(key = lambda x:x.len, reverse=True)
    ambigious_set = convertStrand2Stroke(strands_set,img_width,img_height)
    # 按照 stroke 长度从大到小排序
    ambigious_set.sort(key=lambda x: x.len, reverse=True)
    remain_ambigious_set =[]
    resolve_set = strokes
    for i in range(0,len(strokes)):
        strokes[i].stroke_idx = i

    # 算法运行
    threshold = 30
    for i in range(0, len(ambigious_set)):
        ambi_stroke = ambigious_set[i]
        if ambi_stroke.strand_len() < threshold:
            print i,' <strlen ',threshold
            continue
        #寻找邻近的 resolve set
        search_threshod = 70 # 寻找点间距离在70个像素点内的resolve stroke
        finded_stroke =[]
        for i in range(0,len(resolve_set)):
            resolve_stroke = resolve_set[i]
            cur_dis = caculate_stroke_distance(ambi_stroke,resolve_stroke)
            if cur_dis < search_threshod:
                finded_stroke.append(resolve_stroke)
        print 'len(resolve_set)',len(resolve_set),'finded_stroke',len(finded_stroke)
        # 计算 stroke 间的相似性，得到评分最高的stroke
        if 0: # 这种计算方法不大合理
            min_dis = 100000
            min_stroke = None
            for i in range(0,len(resolve_set)):
                resolve_stroke = resolve_set[i]
                cur_dis = caculate_stroke_distance(ambi_stroke,resolve_stroke)
                if cur_dis < min_dis:
                    min_dis = cur_dis
                    min_stroke = resolve_stroke
            if min_stroke:
                print min_dis,min_stroke
            else:
                print 'can not find resolve stroke'
                continue
        #只使用一定范围内的resolve stroke
        min_score = -1000
        most_similar_stroke = None
        revert_stroke = False
        for i in range(0, len(finded_stroke)):
            resolve_stroke = finded_stroke[i]
            cur_score,should_revert_ambi_dir = caculate_stroke_similarity(ambi_stroke, resolve_stroke)
            if cur_score > min_score:
                min_score = cur_score
                most_similar_stroke = resolve_stroke
                revert_stroke = should_revert_ambi_dir
        if min_score < 0:
            print 'score is smaller than 0',min_score
        new_resolve_stroke = correct_stroke_dir(ambi_stroke,most_similar_stroke,revert_stroke)
        new_resolve_stroke.stroke_idx = len(resolve_set)
        resolve_set.append(new_resolve_stroke)
    return resolve_set
#均为左下角坐标
def render_stroke_to_image(stroke,img,start_end_color = False):
    for i in range(0,stroke.stroke.shape[0]-1):
        x0,y0 = stroke.stroke[i,:]
        x1, y1 = stroke.stroke[i+1, :]
        line_pixel = get_move_line(x0,y0,x1, y1)
        rad = stroke.intrinsic_direction[i]
        rad_color = strand_convert.get_color_from_rad(rad)
        for y,x in line_pixel:
            img[y,x,0:3] = rad_color[::-1] #转化到bgr
    #起点和终点放上不同颜色
    if start_end_color:
        x, y = stroke.stroke[0, :]
        x = int(x)
        y = int(y)
        img[y, x, 0:3] = [255,255,255]
        x, y = stroke.stroke[stroke.stroke.shape[0]-1, :]
        x = int(x)
        y = int(y)
        img[y, x, 0:3] = [127,127,127]

    return img

def resolve_hair_pixel_direction(resolve_set,strock_dir_img,output_dir):
    import cv2
    hair_img = np.zeros((strock_dir_img.shape[0],strock_dir_img.shape[1],3),np.uint8)
    for i in range(0,len(resolve_set)):
        hair_img = render_stroke_to_image(resolve_set[i],hair_img,start_end_color=True)
    hair_img = hair_img[::-1,:,:]
    cv2.imwrite(output_dir+'resove_img.png',hair_img)
    hair_img = np.zeros((strock_dir_img.shape[0],strock_dir_img.shape[1],3),np.uint8)
    smooth_num = 100
    for i in range(0,len(resolve_set)):
        resolve_set[i].lapcain_smooth(iter_num =smooth_num)
        hair_img = render_stroke_to_image(resolve_set[i],hair_img,start_end_color=True)
    hair_img = hair_img[::-1,:,:]
    cv2.imwrite(output_dir + 'resove_img_smooth_'+str(smooth_num)+'.png', hair_img)

#     cur_indx - next3_idx
#         |    \        |
#     next_idx - next2_idx
def convert_img_2_mesh(img,img_bool):
    #假设输入是bool 类型
    height = img.shape[0]
    width = img.shape[1]
    if img.shape[0] != img_bool.shape[0] or img.shape[1] != img_bool.shape[1]:
        print 'img shape not consitent'
    valid_pixels = np.zeros((height,width),dtype=bool)
    face = []
    vertex = []#np.zeros((height*width,3))
    new_corr = np.zeros(height*width)
    count = 0
    for j in range(0,height):
        for i in range(0,width):
            # if i == width -1 or j == height-1:
            #     new_corr[j*width+i] = count
            #     count+=1
            #     if valid_pixels[j,i]:
            #         pass
            # else:
            if j == (height -1) or i ==(width-1):
                new_corr[j*width+i] = count
                valid_pixels [j,i] = True
                count+=1
            else:
                def check_triange_valid(vtx1,vtx2,vtx3):
                    vaild = True
                    return vaild
                if img_bool[j,i]:
                    if img_bool[j+1,i] and img_bool[j+1,i+1] and img_bool[j,i+1]:
                        valid_pixels[j,i] = True
                        new_corr[j * width + i] = count
                        count += 1
                        continue

                        cur_vertex = [i,height-1-j,0] #深度认为是0
                        next_vertex = [i,height-1-(j+1),0]
                        next_vertex2 = [i+1,height-1-(j+1),0]
                        next_vertex3 = [i+1,height-1-j,0]
                        isvalid1 = check_triange_valid(cur_vertex,next_vertex,next_vertex2)
                        isvalid2 = check_triange_valid(cur_vertex, next_vertex2, next_vertex3)
                        if isvalid1 and isvalid2:
                            new_corr[j*width+i] = count
                            valid_pixels[j] = 1
                            count+=1
    count = 0
    for j in range(0,height):
        for i in range(0,width):
            if i == width -1 or j == height-1:
                vertex.append([i, height-1-j, 0])
                continue
                # if valid_pixels[j,i]:
                #     vertex[new_corr[j*width+i]] = [i,height-1-j,0]
                #     continue
            if valid_pixels[j,i]:
                #vertex[new_corr[j * width + i]] = [i, height-1-j, 0]
                vertex.append([i, height-1-j, 0])
                if valid_pixels[j+1,i] and valid_pixels[j+1,i+1] and valid_pixels[j,i+1]:
                    cur_index = j* width +i
                    next_idx = (j+1)*width + i
                    next2_idx = (j+1)*width+i+1
                    next3_idx = j*width+i+1
                    face.append([new_corr[cur_index],new_corr[next_idx],new_corr[next2_idx]])
                    face.append([new_corr[cur_index],new_corr[next2_idx],new_corr[next3_idx]])
    return  vertex, face



def test_img_mesh():
    img_mask = cv2.imread('E:/workspace/dataset/hairstyles/2d_hair/result/A1301043736980A/'+'img_mask_.png',cv2.IMREAD_UNCHANGED)
    img_mask = cv2.imread('E:/workspace/dataset/hairstyles/2d_hair/result/A13010436691206/' + 'img_mask_.png',
                          cv2.IMREAD_UNCHANGED)
    valid_img = np.zeros((img_mask.shape[0],img_mask.shape[1]),dtype=np.bool)
    for j in range(0,img_mask.shape[0]):
        for i in range(0,img_mask.shape[1]):
            if img_mask[j,i] >1:
                valid_img[j, i] = True
            else:
                valid_img[j,i] = False

    V,F,C = convert_img_2_mseh_new(img_mask,valid_img)
    V = np.array(V)
    F = np.array(F)
    C = np.array(C)
    # write_full_obj(V,F,np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),
    #                'E:/workspace/dataset/hairstyles/2d_hair/result/A1301043736980A/img_mask_.obj')
    write_full_obj(V,F,np.array([]),np.array([]),np.array([]),np.array([]),C,
                   'E:/workspace/dataset/hairstyles/2d_hair/result/A13010436691206/img_mask_new.obj')

def test_mapping():
    from fitting.util import  read_igl_obj,boudaray_loop,map_vertices_to_circle,Harmonic
    #v, f, t, t_f, n, n_f = read_igl_obj('E:/workspace/dataset/hairstyles/2d_hair/result/A1301043736980A/'+'img_mask_2_cluster_simplification.obj')
    v, f, t, t_f, n, n_f = read_igl_obj(
        'E:/workspace/dataset/hairstyles/2d_hair/result/A13010436691206/' + 'img_mask_simple.obj')

    mean_v = np.mean(v,axis=0)
    bnd = boudaray_loop(f) #获得边界点的index
    color_vertex = np.zeros((v.shape[0],3),np.uint8)
    color_vertex[:,:] = [1,1,1]
    for i in range(0,bnd.shape[0]):
        index = bnd[i,0]
        corr_vertex = v[index,:]
        dir_3d = corr_vertex - mean_v
        dir_3d = dir_3d/np.linalg.norm(dir_3d)
        dir_2d = dir_3d[0:2]

        #rgb = strand_convert.get_color_from2d_dir(dir_2d)
        cur_rad = float(i)/bnd.shape[0]*2*pi
        rgb = strand_convert.get_color_from_rad(cur_rad)
        color_vertex[index,:] = rgb
    color_vertex[bnd[0,0], :] = [0,0,0] #起点颜色
    # write_full_obj(v,f,np.array([]),np.array([]),np.array([]),np.array([]),color_vertex,
    #                'E:/workspace/dataset/hairstyles/2d_hair/result/A1301043736980A/img_mask_boundary_color.obj')
    write_full_obj(v,f,np.array([]),np.array([]),np.array([]),np.array([]),color_vertex,
                   'E:/workspace/dataset/hairstyles/2d_hair/result/A13010436691206/img_mask_boundary_color.obj')
    bnd_uv = map_vertices_to_circle(v,bnd)
    v_uv = Harmonic(v,f,bnd,bnd_uv,1)
    V = np.zeros((v_uv.shape[0],3))

    for i in range(0,v_uv.shape[0]):
        V[i,0:2] = v_uv[i,:]
        V[i,2] = 0

    # write_full_obj(V,f,np.array([]),np.array([]),np.array([]),np.array([]),color_vertex,
    #                'E:/workspace/dataset/hairstyles/2d_hair/result/A1301043736980A/img_mask_remap.obj')
    write_full_obj(V,f,np.array([]),np.array([]),np.array([]),np.array([]),color_vertex,
                   'E:/workspace/dataset/hairstyles/2d_hair/result/A13010436691206/img_mask_remap.obj')

def test_2_mesh_mapping(path1,path2,out_path1,out_path2):
    from fitting.util import read_igl_obj, boudaray_loop, map_vertices_to_circle, Harmonic,\
        ARAP_DATA,arap_precomputation,arap_solve
    v1, f1, t1, t_f1, n1, n_f1 = read_igl_obj(path1)
    v2, f2, t2, t_f2, n2, n_f2 = read_igl_obj(path2)
    bnd1 = boudaray_loop(f1) #获得边界点的index
    bnd2 = boudaray_loop(f2)  # 获得边界点的index
    if 0:
        new_index = np.linspace(0,bnd1.shape[0]-1,bnd1.shape[0]/5,dtype=int)
        bnd1 = bnd1[new_index,:]
    print 'bnd1 size',bnd1.size
    print 'bnd2 size',bnd2.size
    color_vertex1 = np.zeros((v1.shape[0],3),np.uint8)
    color_vertex1[:,:] = [1,1,1]
    color_vertex2 = np.zeros((v2.shape[0],3),np.uint8)
    color_vertex2[:,:] = [1,1,1]

    from sklearn.neighbors import NearestNeighbors
    source_mesh  = v1[bnd1[:,0]]
    target_mesh =  v2[bnd2[:,0]]
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(target_mesh)
    distances, indices = neigh.kneighbors(source_mesh, return_distance=True)
    bnd_uv = np.zeros((source_mesh.size,3))
    for i in range(0,indices.shape[0]):
        bnd_uv[i,:] = target_mesh[indices[i]]
    v_uv = Harmonic(v1,f1,bnd1,bnd_uv,1)
    arap_data = ARAP_DATA(0)
    b = np.array([])
    bc =  np.array([])
    arap_precomputation(v1,f1,2,b,arap_data.data)
    arap_uv = arap_solve(bc,arap_data.data,v_uv)


    write_full_obj(arap_uv,f1,np.array([]),np.array([]),np.array([]),np.array([]),color_vertex1,
                   out_path1)
    return

    for i in range(0,bnd1.shape[0]):
        index = bnd1[i,0]

        #rgb = strand_convert.get_color_from2d_dir(dir_2d)
        cur_rad = float(i)/bnd1.shape[0]*2*pi
        rgb = strand_convert.get_color_from_rad(cur_rad)
        color_vertex1[index,:] = rgb
    color_vertex1[bnd1[0,0], :] = [0,0,0] #起点颜色
    write_full_obj(v1,f1,np.array([]),np.array([]),np.array([]),np.array([]),color_vertex1,
                   out_path1)
    for i in range(0,bnd2.shape[0]):
        index = bnd2[i,0]
        #rgb = strand_convert.get_color_from2d_dir(dir_2d)
        cur_rad = float(i)/bnd2.shape[0]*2*pi
        rgb = strand_convert.get_color_from_rad(cur_rad)
        color_vertex2[index,:] = rgb
    color_vertex2[bnd2[0,0], :] = [0,0,0] #起点颜色
    write_full_obj(v2,f2,np.array([]),np.array([]),np.array([]),np.array([]),color_vertex2,
                   out_path2)

if __name__ == '__main__':
    #input_dir = 'G:/yuanqing/faceproject/hairstyles/'
    input_dir = 'E:/workspace/dataset/hairstyles/2d_hair/'
    from fitting.util import genertate_dir_color,read_strand_txt_file
    import strand_convert
    #genertate_dir_color(1024,1024,input_dir+'undirected_color.png',getcolor_from_rad)
    #genertate_dir_color(1024,1024, input_dir + 'directed_color.png', strand_convert.get_color_from_rad)
#    get_orientation_map(input_dir+'hair1.png',input_dir+'orientaion_map_grey_cv/')
    #get_orientation_map_cv_new(input_dir+'coeff_map_ksize60sigma_3lambda_4.0gamma_0.5psi_0.png',input_dir+'refine/')
#    get_orientation_map_cv_new(input_dir+'hair2.png',input_dir+'hair2_1/')
    #get_orientation_map_cv_new(input_dir + 'coeff_map_ksize50sigma_2lambda_4.0gamma_0.5psi_0.png', input_dir + 'refine/')
    #get_orientation_map_cv_new(input_dir + 'kk_coe_weight.png', input_dir + 'kk_weight/')
    #get_orientation_map_cv_new(input_dir+'coeff_map.png',input_dir+'coeff/')

    #strand_trace(input_dir+'/1/'+'init_strand_20.0ksize50sigma_2.0lambda_4.0gamma_0.5psi_0.pkl',
    #             input_dir + '/1/'+'extract/')
#    strand_trace(input_dir + '/hair2_1/' + 'init_strand_20.0ksize50sigma_2.0lambda_4.0gamma_0.5psi_0.pkl',input_dir + '/hair2_1/'+'extract/')

    #get_orientation_map_cv_new(input_dir+'5.png',input_dir+'5/')
    #strand_trace(input_dir + '/5/' + 'init_strand_20.0ksize50sigma_2.0lambda_4.0gamma_0.5psi_0.pkl',
    #             input_dir + '/5/' + 'extract/')

    #test_img_mesh()
    #test_mapping()
    #test_img_mesh()
    if 0:
        indir = 'E:/workspace/dataset/hairstyles/2d_hair/test/'
        test_2_mesh_mapping(indir+'source_cut_smooth.obj',indir+'img_mask_target_smooth_subdiv.obj',
                            indir+'source_remap_arap_1.obj',indir+'target_color.obj'
                            )


    if 0:
        buk = ['A1301043678290A','A1301043736980A']
        buk = ['A13010437115307','A13010438114107']
        #buk = ['A13010438114107']
        buk = ['A1301043678290A', 'A1301043736980A','A1301043828500A','A13010436665609','A13010436691206','A13010436840808',
               'A13010436856307','A13010436892702','A13010436944105','A13010437417908',
               'A13010437446107','A13010437615200','A13010437672005']
        format ='.jpg'
        for i in range(0,len(buk)):
            out_dir =  input_dir+'result/'+buk[i]+'/'
            safe_mkdirs(out_dir)
            input_img_path = input_dir + buk[i]+ format
            input_seg_img_path = input_dir + 'Seg_refined/'+ buk[i]+'.png'
            input_strand_img_path = input_dir + 'Strand/' + buk[i] + '.png'
            input_strand_file_path = input_dir + 'Strand/' + buk[i] + '.txt'
            strock_dir_img = cv2.imread(input_strand_img_path, cv2.IMREAD_COLOR)
            hair_array = read_strand_txt_file(input_strand_file_path)
            if 1:
                get_orientation_map_cv_new(input_img_path,input_seg_img_path, out_dir, 1)
            extract_dir = out_dir+ 'extract/'
            safe_mkdirs(extract_dir)
            strands_set = strand_trace(out_dir + 'init_strand_20.0ksize50sigma_2.0lambda_4.0gamma_0.5psi_0.pkl',
                         extract_dir)
            resolve_dir = out_dir+ 'resolve/'
            safe_mkdirs(resolve_dir)
            print 'start resolve strand direction'
            if not os.path.exists(resolve_dir+'resolve_result.pkl'):
                resolve_result = resolve_strand_direction(strands_set,strock_dir_img,hair_array)
                save_binary_pickle(resolve_result,resolve_dir+'resolve_result.pkl')
            else:
                print 'resolve_result.pkl exist '
                resolve_result = load_binary_pickle(resolve_dir+'resolve_result.pkl')
            print 'end resolve strand direction'
            print 'start resolve_hair_pixel_direction'
            resolve_hair_pixel_direction(resolve_result,strock_dir_img,resolve_dir)
            print 'end resolve_hair_pixel_direction'

    # get_orientation_map_cv_new(input_dir+'2.png',input_dir+'2/',1)
    #
    # strand_trace(input_dir + '/2/' + 'init_strand_20.0ksize50sigma_2.0lambda_4.0gamma_0.5psi_0.pkl',
    #              input_dir + '/2/' + 'extract/')



