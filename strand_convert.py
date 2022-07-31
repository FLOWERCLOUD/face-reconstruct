# -- coding: utf-8 --
import array
import numpy as np
from numba import jit
from fitting.util import write_simple_obj,add_vertex_faces,FileFilt,write_full_obj,readVertexColor,write_landmark_to_obj,\
    save_binary_pickle,load_binary_pickle,safe_mkdirs,read_igl_obj,get_vertex_normal
import  math
from math import sin, cos,atan,pi
from time import time
import sys


def print_para_resut(step, timer_end, timer_start):
    print step
    print "in %f sec\n" % (timer_end - timer_start)
def read_bin(filename):
    result =[]
    try:
        with open(filename, 'rb') as f:
            a = array.array("L")  # L is the typecode for uint32
            a.fromfile(f, 1)
            print a
            for i in range(0, a[-1]):
                a.fromfile(f, 1)
                # print a
                strand = []
                if a[-1] == 1:
                    #print i,a[-1]
                    pass
                for j in range(0, a[-1]):
                    b = array.array("f")
                    b.fromfile(f, 3)
                    point = []
                    point.append(b[0])
                    point.append(b[1])
                    point.append(b[2])
                    strand.append(point)
                # print b
                result.append(strand)
        return np.array(result)
    except IOError:
        return np.array(result)

'''
Et = (cos(phi)cos(theta),cos(phi)sin(theta),sin(phi))

'''
#@jit(nopython=True, cache=True)
def convert_to_spherical_tangent(x,y,z,prev_phi,prev_theta,is_first = False):
    phi = math.asin(z)
    if( abs(x) >=0.00000001):
        tan_theta = y/x
        theta = math.atan(tan_theta)

        if is_first:
            pass
        else:
            if abs( theta- prev_theta)>math.pi/2:
                if theta > prev_theta:
                    theta-= math.pi
                else:
                    theta += math.pi
        cos_phi = x / math.cos(theta)
        if cos_phi* cos(phi)<0:
            phi = math.pi - phi
            if phi >= 2*math.pi:
                phi -= 2*math.pi
        if cos_phi*cos(theta)*x <0:
            theta+=math.pi
            if theta > 2*math.pi:
                theta - 2*math.pi
        #if cos_phi * z < 0:
         #   phi += math.pi
        #print cos(phi)*cos(theta),cos(phi)*sin(theta),sin(phi)
        return phi, theta
    else:
        print y, x
        if y*x >0:
            theta = math.pi/2
        else:
            theta = -math.pi / 2
        if is_first:
            pass
        else:
            if abs( theta- prev_theta)>math.pi/2:
                if theta > prev_theta:
                    theta-= math.pi
                else:
                    theta += math.pi
        cos_phi = y / math.sin(theta)
        if cos_phi* cos(phi)<0:
            phi = math.pi - phi
            if phi >= 2*math.pi:
                phi -= 2*math.pi
        if cos_phi*cos(theta)*x <0:
            theta+=math.pi
            if theta > 2*math.pi:
                theta - 2*math.pi
        print phi, theta
        return phi, theta

#@jit(nopython=True, cache=True)
def convert_to_helix(strand):
    phis= []
    thetas=[]
    points =[]

    if len(strand) == 1:
        #print 'len(strand)',len(strand)
        return phis,thetas,points
#    strand[:,1] -=1.7178 # 归一化数值
    for i in range(0, len(strand)-1): #最后一个点的tangent 来源于前一个
        tangent = strand[i+1][:] - strand[i][:]
        tangent/= np.linalg.norm(tangent)
        if i == 0:
            phi,theta = convert_to_spherical_tangent(tangent[0],tangent[1],tangent[2],0,0,is_first =True)
        else:
            phi, theta = convert_to_spherical_tangent(tangent[0], tangent[1], tangent[2], phis[-1], thetas[-1])

        phis.append(phi)
        thetas.append(theta)
        points.append(strand[i][:])
    phis.append(phis[-1])
    thetas.append(thetas[-1])
    points.append(points[-1])
    return  phis,thetas,points

def cacaulate_rot(stran_dir,Et,En,Eb):
    if check_if_contained_nan(stran_dir):
        print stran_dir
#    print 'cacaulate_rot',stran_dir,Et,En,Eb
    # 因为已经归一化到中心，因此点的位置可以点应具有的朝向
    # 使用第一个点的位置作为方向
    # stran_dir = strand[0][:] / np.linalg.norm(strand[0][:])
    # Et = strand[1][:] - strand[0][:]
    # Et /= np.linalg.norm(Et)
    # En = np.array([-sin(thetas[0]), cos(thetas[0]), 0])
    # Eb = np.array([-sin(phis[0]) * cos(thetas[0]), -sin(phis[0]) * sin(thetas[0]), cos(phis[0])])

    # 把 stran_dir 投影到 En，Eb 平面
    m = np.vstack([Et, En, Eb])  # 投射矩阵
    Et = Et.reshape(Et.size, 1)
    En = En.reshape(En.size, 1)
    Eb = Eb.reshape(Eb.size, 1)
    m2 = np.concatenate((Et, En, Eb), axis=1)
    Et = Et[:, 0]
    En = En[:, 0]
    Eb = Eb[:, 0]
    project_coff = np.dot(m, stran_dir)  # 使用 .dot才是矩阵乘法，否则是对应相乘
    project_coff[0] = 0  # 投影到 n,b平面
    cast_stran_dir = m2.dot(project_coff)
#    print 'cast_stran_dir', cast_stran_dir
    dot_resut = np.dot(Eb, cast_stran_dir) / np.linalg.norm(Eb) / np.linalg.norm(cast_stran_dir)
#    print 'dot_resut',dot_resut
#    dot_resut = np.clip(dot_resut,-0.9999,0.9999)
    rot_theta = 0
    if (dot_resut > 0.999):
        pass
    # elif (abs(dot_resut) <0.0001):
    #     pass
    else:
        dot_resut = np.clip(dot_resut, -0.999, 0.999)
        rot_theta = math.acos(dot_resut)
        cross_result = np.cross(Eb, cast_stran_dir)
        if np.dot(cross_result, Et) < 0:
            # 说明角度应小于180度
            pass
        else:
            # 说明角度应大于180度
            if rot_theta < math.pi:
                rot_theta = 2 * math.pi - rot_theta
    return rot_theta

def get_rad_from2d_dir(dir):
    from math import pi
    if np.linalg.norm(dir) < 0.00001:
        print 'get_rad_from2d_dir wrong,norm',np.linalg.norm(dir)
    dir/=np.linalg.norm(dir)
    dir = dir[0:2] #cast 2d
    if np.linalg.norm(dir) <0.0001:
        return [255,255,255]
    else:
        dir/= np.linalg.norm(dir)
    x = dir[0]
    y = dir[1]
    #我们以[0,1] 作为起点轴，顺时针旋转
    if abs(y) <0.0001:
        if x >0:
            theta = pi/2
        else:
            theta = 3*pi/2
    else:
        theta = atan(x/y)
        if theta >= 0:
            if y>0 and x>0 :
                pass
            else: # y<0 and x <0
                theta +=pi
        else:
            if y<0 and x >0:
                theta +=pi
            else: #y>0 ,x<0
                theta +=2*pi
    return theta

def get_color_from_rad(rad):
    # 颜色排列 rgb
    theta = rad
    theta_range = [0,pi/3,pi/3*2,pi,pi/3*4,pi/3*5,2*pi]
    color_range = np.array([[0,255,255],[0,255,0],[255,255,0],[255,0,0],[255,0,255],[0,0,255]])
    theta_step = pi/3
    if theta <0:
        theta = 0
    if theta> 2*pi:
        theta = 2*pi

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
def get_color_from2d_dir(dir):

    if np.linalg.norm(dir) < 0.0001:
        print 'dir wrong'
    dir/=np.linalg.norm(dir)
    dir = dir[0:2] #cast 2d
    if np.linalg.norm(dir) <0.0001:
        return [255,255,255]
    else:
        dir/= np.linalg.norm(dir)
    x = dir[0]
    y = dir[1]
    #我们以[0,1] 作为起点轴，顺时针旋转
    if abs(y) <0.0001:
        if x >0:
            theta = pi/2
        else:
            theta = 3*pi/2
    else:
        theta = atan(x/y)
        if theta >= 0:
            if y>0 and x>=0 :
                pass
            else: # y<0 and x <0
                theta +=pi
        else:
            if y<0 and x >=0:
                theta +=pi
            else: #y>0 ,x<0
                theta +=2*pi
    return  get_color_from_rad(theta)

def get_rad_from_color(color):

    theta_range = [0,pi/3,pi/3*2,pi,pi/3*4,pi/3*5,2*pi]
    color_range = np.array([[0,255,255],[0,255,0],[255,255,0],[255,0,0],[255,0,255],[0,0,255]])

    r = color[0]
    g = color[1]
    b = color[2]
    range1 = 0
    # 因为出现情况 ，显示是255.0 单时间比255.0 小，导致不等于 255,因此这里用round 四舍五入处理了一下
    if  r == 0:

        #print b,round(b),type(round(b))
        if round(g) ==255:
            range1 =0
        elif round(b) ==255:
            range1 = 5
    elif g == 0:
        if round(r) == 255:
            range1 = 3
        elif round(b) ==255:
            range1 = 4
    elif b == 0:
        if round(r) == 255:
            range1 = 2
        elif round(g) == 255:
            range1 = 1
    base_rad = pi / 3 * range1
    theta_step = pi/3
    if range1 ==0:
        x1 =color_range[range1]
        x2 = color_range[range1+1]
    elif range1 ==1:
        x1 =color_range[range1]
        x2 = color_range[range1+1]
    elif range1 ==2:
        x1 =color_range[range1]
        x2 = color_range[range1+1]
    elif range1 ==3:
        x1 =color_range[range1]
        x2 = color_range[range1+1]
    elif range1 ==4:
        x1 =color_range[range1]
        x2 = color_range[range1+1]
    elif range1 ==5:
        x1 =color_range[range1]
        x2 = color_range[0]
    #print range1,color-x1,x2-x1
    p1 = color - x1
    p2 = x2-x1
    for i in range(0,3):
        if p2[i] == 0:
            continue
        else:
            p = p1[i]/p2[i]
            break
    target_rad = base_rad + theta_step*p
    return  target_rad


def convert_2_mesh(phis,thetas,points ,constraint_dir =np.array([]), radius = 0.005,step =9,if_fix_dir = False,g_div=2):

    phis = np.array(phis)
    thetas = np.array(thetas)
    points =  np.array(points)
    v =[]
    f =[]
    vt =[]
    vt_f =[]
    v_color=[]
    if g_div == 6:
        angles = [0.0,2*math.pi/6,2*math.pi/3,math.pi,4*math.pi/3,10*math.pi/6]
    elif g_div ==2:
        angles = [0.0, math.pi]
    m_range = range(0,points.shape[0],step)

    for index ,i in enumerate(m_range):
        ratio = float(index)/(len(m_range)-1)
        # if i > 10:
        #     break
        phi = phis[i]
        theata = thetas[i]
        node = np.array(points[i])
        #print 'constraint_dir',constraint_dir
        if constraint_dir.size > 0 and not check_if_contained_nan(constraint_dir[i,:]):
            constraint_dir[i,:]/= np.linalg.norm(constraint_dir[i,:])
            stran_dir = constraint_dir[i,:]
        else:
            stran_dir = node / np.linalg.norm(node)
        Et  = np.array([cos(phi)*cos(theata),cos(phi)*sin(theata),sin(phi)])
        En = np.array([-sin(theata),cos(theata),0])
        Eb = np.array([-sin(phi)*cos(theata),-sin(phi)*sin(theata),cos(phi)])
        if if_fix_dir: #对旋转方向进行修正
            rot = cacaulate_rot(stran_dir, Et, En, Eb)
        else:
            rot = 0

        for rad in angles:
            rad -=rot # 对角度进行修正
            v_c = node+(cos(rad)* En + sin(rad)*Eb)*radius
            v.append(v_c)
        if g_div == 2:
            # vt_tmp =[1.0, 1.0-ratio]
            # vt.append(vt_tmp)
            # vt_tmp =[0.8, 1.0-ratio]
            # vt.append(vt_tmp)
            vt_tmp =[0.22, 0.9*(1-ratio)]
            vt.append(vt_tmp)
            vt_tmp =[0.0, 0.9*(1-ratio)]
            vt.append(vt_tmp)
        if i ==0:

            if g_div == 6:
                v0 = 0
                v1 = 1
                v2 = 2
                v3 = 3
                v4 = 4
                v5 = 5
                f.append([v0, v5, v1])
                f.append([v5, v4, v1])
                f.append([v4, v3, v1])
                f.append([v3, v2, v1])
            elif g_div == 2:
                v0 = 0
                v1 = 1
                v_color.append(get_color_from2d_dir(Et))
                v_color.append(get_color_from2d_dir(Et))

            continue
        else:
            if g_div == 6:
                v6 = v0 + 6
                v7 = v1 + 6
                v8 = v2 + 6
                v9 = v3 + 6
                v10 = v4 + 6
                v11 = v5 + 6
                f.append([v0, v1, v7])
                f.append([v0, v7, v6])
                f.append([v5, v0, v6])
                f.append([v5, v6, v11])
                f.append([v4, v5, v11])
                f.append([v4, v11, v10])
                f.append([v3, v4, v10])
                f.append([v3, v10, v9])
                f.append([v2, v3, v9])
                f.append([v2, v9, v8])
                f.append([v1, v2, v8])
                f.append([v1, v8, v7])
                v0 = v6
                v1 = v7
                v2 = v8
                v3 = v9
                v4 = v10
                v5 = v11
            elif g_div == 2:
                v2 = v0+2
                v3 = v1+2
                f.append([v0, v1, v2])
                f.append([v2, v1, v3])
                vt_f.append([v0, v1, v2])
                vt_f.append([v2, v1, v3])
                v0 = v2
                v1 = v3
                v_color.append(get_color_from2d_dir(Et))
                v_color.append(get_color_from2d_dir(Et))


    if index>0 and g_div>2:
        f.append([v0,v1,v5])
        f.append([v1,v2,v5])
        f.append([v2,v3,v5])
        f.append([v3,v4,v5])
    return np.array(v),np.array(f),np.array(vt),np.array(vt_f),np.array(v_color)
#@jit(nopython=True, cache=True)
def convert_batch(result,num = 100,if_fix_dir = False):
    mesh_v =[]
    mesh_f =[]
    mesh_v = np.array(mesh_v)
    mesh_f = np.array(mesh_f)
    mesh_n =[]
    mesh_n_f =[]
    mesh_vt = []
    mesh_vt_f =[]
    mesh_n = np.array(mesh_n)
    mesh_n_f = np.array(mesh_n_f)
    mesh_vt = np.array(mesh_vt)
    mesh_vt_f = np.array(mesh_vt_f)
    mesh_v_color = np.array([])

    step = result.shape[0]/num
    if step < 1:
        step =1
    for i in range(0,result.shape[0],step):
        #print i
        phis,thetas,points = convert_to_helix(np.array(result[i]))
        if len(phis) == 0:
            continue
        #print phis
        #print thetas
        v,f,vt,vt_f,v_color = convert_2_mesh(phis,thetas,points, radius = 0.01,step =4,if_fix_dir=if_fix_dir)
        #mesh_v.append(v)
        #mesh_f.append(f)
        mesh_v,mesh_f = add_vertex_faces(mesh_v,mesh_f,v,f)
        mesh_vt ,mesh_vt_f = add_vertex_faces(mesh_vt,mesh_vt_f,vt,vt_f)
        if mesh_v_color.size == 0:
            mesh_v_color = v_color
        else:
            mesh_v_color = np.vstack((mesh_v_color , v_color))
    return  mesh_v,mesh_f,mesh_n,mesh_n_f,mesh_vt,mesh_vt_f,mesh_v_color



def convert_batch_with_label(result,label,label_num,num = 100,if_fix_dir = False,select_label_num =1):
    mesh_v =[]
    mesh_f =[]
    mesh_v = np.array(mesh_v)
    mesh_f = np.array(mesh_f)
    mesh_n =[]
    mesh_n_f =[]
    mesh_vt = []
    mesh_vt_f =[]
    mesh_n = np.array(mesh_n)
    mesh_n_f = np.array(mesh_n_f)
    mesh_vt = np.array(mesh_vt)
    mesh_vt_f = np.array(mesh_vt_f)
    mesh_v_color = np.array([])

    step = result.shape[0]/num
    if step < 1:
        step =1
    if select_label_num >0:
        step =1
    label_count = np.zeros(label_num,np.int)
    for i in range(0,result.shape[0],step):
        #print i
        phis,thetas,points = convert_to_helix(np.array(result[i]))
        if len(phis) == 0:
            continue
        #print phis
        #print thetas
        cur_label = label[i]
        if label_count[cur_label] >= select_label_num:
            continue
        else:
            label_count[cur_label]+=1
        cur_color = get_color_from_rad(2*pi*cur_label/label_num)
        v,f,vt,vt_f,v_color = convert_2_mesh(phis,thetas,points, radius = 0.01,step =4,if_fix_dir=if_fix_dir)
        v_color = np.zeros((v.shape[0],3))
        v_color[:,:] = cur_color
        #mesh_v.append(v)
        #mesh_f.append(f)
        mesh_v,mesh_f = add_vertex_faces(mesh_v,mesh_f,v,f)
        mesh_vt ,mesh_vt_f = add_vertex_faces(mesh_vt,mesh_vt_f,vt,vt_f)
        if mesh_v_color.size == 0:
            mesh_v_color = v_color
        else:
            mesh_v_color = np.vstack((mesh_v_color , v_color))
    return  mesh_v,mesh_f,mesh_n,mesh_n_f,mesh_vt,mesh_vt_f,mesh_v_color

@jit(nopython=True, cache=True)
def test():
    result =[]
    for i in range(0,10000):
        result.append(i)
    result.sort(reverse=True)
    return result

#result = test()
#print result
# result = read_bin(file_dir+'strands00002.data')
# result = np.array(result)
# t1 =time()
# #phis,thetas= convert_to_helix(np.array(result[0]))
# stand_num =10000
# v ,f=convert_batch(result,num = stand_num)
# #v[:,1] -=1.7178
# print_para_resut('convert_to_helix', time(), t1)

def generate_polystrip_mesh():
    file_dir = "G:/yuanqing/faceproject/hairstyles/hairstyles/"
    prj_dir = 'G:/yuanqing/faceproject/hairstyles/hairstyles/hair/'
    b = FileFilt()
    b.FindFile(dirr=prj_dir)
    count =0
    target = 1
    for k in b.fileList:
        if k == '':
            continue
        if count == target:
            pass
        else:
            count += 1
            continue
        print k.split("/")[-1]
        filename_split = k.split("/")[-1].split(".")
        print filename_split
        if len(filename_split) > 1:
            #       print str(filename_split[-2])
            file_name = str(filename_split[-2])
        else:
            file_name = str(filename_split[0])

        result = read_bin(prj_dir +  file_name + '.data')
        stand_num = 1000

        v, f,n,nf,vt,vt_f = convert_batch(result, num=stand_num)
        write_full_obj(v, f, n, nf, vt, vt_f, np.array([]), prj_dir+'/convert_hair_texture/'+file_name+'_'+str(stand_num)+'.obj', generate_mtl=True,
                       verbose=False, img_name='hair2.png')
        #write_simple_obj(v,f,prj_dir+'/convert_hair/'+file_name+'_'+str(stand_num)+'.obj')
        count+=1
def generate_polystrip_mesh_with_dir_color():
    file_dir = 'E:/workspace/dataset/hairstyles/'
    prj_dir = 'E:/workspace/dataset/hairstyles/hair/'
    b = FileFilt()
    b.FindFile(dirr=prj_dir)
    count =0
    target = 1
    for k in b.fileList:
        if k == '':
            continue
        if count == target:
            pass
        else:
            count += 1
            #continue
        print k.split("/")[-1]
        filename_split = k.split("/")[-1].split(".")
        print filename_split
        if len(filename_split) > 1:
            #       print str(filename_split[-2])
            file_name = str(filename_split[-2])
        else:
            file_name = str(filename_split[0])

        result = read_bin(prj_dir +  file_name + '.data')
        stand_num = 1000

        v, f,n,nf,vt,vt_f,vt_color = convert_batch(result, num=stand_num,if_fix_dir=False)
        if 0:
            write_full_obj(v, f, n, nf, vt, vt_f, vt_color, prj_dir+'/convert_hair_dir_new/'+file_name+'_'+str(stand_num)+'.obj', generate_mtl=False,
                       verbose=False, img_name='hair2.png')
        elif 0:
            write_full_obj(v, f, n, nf, vt, vt_f, vt_color, prj_dir+'/convert_hair_no_constraint/'+file_name+'_'+str(stand_num)+'.obj', generate_mtl=False,
                       verbose=False, img_name='hair2.png')
        elif 1:
            write_full_obj(v, f, n, nf, vt, vt_f, vt_color,prj_dir + '/convert_hair_helix/' + file_name + '_' + str(stand_num) + '.obj',generate_mtl=False,
                       verbose=False, img_name='hair2.png')
        else:
            write_full_obj(v, f, n, nf, vt, vt_f, vt_color,
                           prj_dir + '/convert_hair_with_texture/' + file_name + '_' + str(stand_num) + '.obj',
                           generate_mtl=True,
                           verbose=False, img_name='hair_06_Hair_Diffuse_Opacity.png')

        #write_simple_obj(v,f,prj_dir+'/convert_hair/'+file_name+'_'+str(stand_num)+'.obj')

        if count >30 :
            break
        count+=1

def generta_segment_map():
    from z_buffer_raster import Mesh_render_to_image
    import sys
    sys.path.insert(0, "D:/mprojects/libiglfull/libigl/python/build/x64/Release")
    import pyigl as igl
    import numpy as np
    from triangle_raster import MetroMesh
    from fitting.util import  read_igl_obj,add_vertex_faces
    hair_file_dir = "G:/yuanqing/faceproject/hairstyles/hairstyles/"+"hair/convert_hair"
    head_filr_dir = "G:/yuanqing/faceproject/hairstyles/hairstyles/"
    hair_objname = 'strands00002_1000'
    head_objname = 'frame_aligned'
    hair_v,hair_v_f,hair_t_v,hair_t_f,hair_n_v,hair_n_f= read_igl_obj(hair_file_dir + '/' + hair_objname + '.obj')
    hair_v = hair_v[:, 0:3]
    head_v,head_v_f,head_t_v,head_t_f,head_n_v,head_n_f= read_igl_obj(head_filr_dir + '/' + head_objname + '.obj')
    head_v = head_v[:, 0:3]

    hair_vertex_color= np.zeros((hair_v.shape[0],3))
    hair_vertex_color[:,:] = [255,0,0]
    head_vertex_color= np.zeros((head_v.shape[0],3))
    head_vertex_color[:,:] = [0,255,0]

    #merge to one mesh
    merge_v, merge_v_f = add_vertex_faces(hair_v,hair_v_f,head_v,head_v_f)
    merge_t_v, merge_t_f = add_vertex_faces(hair_t_v, hair_t_f, head_t_v, head_t_f)
    merge_n_v, merge_n_f = add_vertex_faces(hair_n_v, hair_n_f, head_n_v, head_n_f)
    merge_color = np.vstack((hair_vertex_color, head_vertex_color))
    mesh = MetroMesh()
    mesh.set_mesh(v=merge_v, vertex_color=merge_color, normal=merge_n_v,
                  vt=merge_t_v, face=merge_v_f, n_face=merge_n_f, t_face=merge_t_f)

    Mesh_render_to_image(hair_file_dir+'/head_has_uv_python_render.png',mesh,256,256)
    pass
def generta_segment_map_batch(output_name,use_vertex_color = False):
    from z_buffer_raster import Mesh_render_to_image,Mesh_render_to_image_withmy_bbox
    import sys
    sys.path.insert(0, "E:/workspace/igl_python/")
    import pyigl as igl
    import numpy as np
    from triangle_raster import MetroMesh
    from fitting.util import  read_igl_obj,add_vertex_faces
    from fitting.landmarks import  load_embedding

    sys.path.insert(0, "D:/mproject/meshlab2016/meshlab/src/x64/Release/")
    import meshlab_python



    hair_file_dir = "E:\workspace/dataset/hairstyles/"+ "hair/convert_hair_dir/"#"hair/convert_hair/"
    head_filr_dir = "E:\workspace/dataset/hairstyles/"
    head_objname = 'frame_aligned'  #''frame_aligned'
    body_objname = 'nohead_body'

    head_v,head_v_f,head_t_v,head_t_f,head_n_v,head_n_f= read_igl_obj(head_filr_dir + '/' + head_objname + '.obj')
    head_v = head_v[:, 0:3]
    body_v,body_v_f,body_t_v,body_t_f,body_n_v,body_n_f= read_igl_obj(head_filr_dir + '/' + body_objname + '.obj')
    body_v = body_v[:, 0:3]

    head_vertex_color= np.zeros((head_v.shape[0],3))
    head_vertex_color[:,:] = [0,255,0]
    body_vertex_color= np.zeros((body_v.shape[0],3))
    body_vertex_color[:,:] = [0,255,0]

    merge_head_body_v, merge_head_body_v_f = add_vertex_faces(head_v, head_v_f, body_v, body_v_f)
    merge_head_body_t_v, merge_head_body_t_f = add_vertex_faces(head_t_v, head_t_f, body_t_v, body_t_f)
    merge_head_body_n_v, merge_head_body_n_f = add_vertex_faces(head_n_v, head_n_f, body_n_v, body_n_f)

    merge_head_body_color = np.vstack((head_vertex_color, body_vertex_color))

    # landmark embedding
    lmk_emb_path = './data/lmk_embedding_intraface_to_flame.pkl'
    lmk_face_idx, lmk_b_coords = load_embedding(lmk_emb_path)
    from  fitting.util import mesh_points_by_barycentric_coordinates
    face_landmark_3d = mesh_points_by_barycentric_coordinates(head_v, head_v_f, lmk_face_idx, lmk_b_coords)
    if 0:
        model_path = './models/male_model.pkl'  # change to 'female_model.pkl' or 'generic_model.pkl', if needed
        from smpl_webuser.serialization import load_model
        model = load_model(model_path)

        #write_simple_obj(model.r,model.f,hair_file_dir + output_name + 'frame' + '.obj')
        write_landmark_to_obj(hair_file_dir + output_name + 'landmark2' + '.obj',face_landmark_3d,size = 3)
    #这里的landmark没有包括下巴轮廓
    #我们计算它的包围盒
    from triangle_raster import BBox3f
    bbox = BBox3f()
    bbox.addvertex_array(face_landmark_3d)

    x_center = (bbox.max[0] + bbox.min[0])/2
    y_center = (bbox.max[1] + bbox.min[1]) / 2
    z_center = (bbox.max[2] + bbox.min[2]) / 2
    x_scale =5
    y_scale_up = 4.0
    y_scale_down = 20.0
    z_scale_back = 20.0
    z_scale_front = 20.0
    #利用x 来估计z
    #bbox.max[2] = z_center +  (bbox.max[0]-  x_center)*z_scale_front
    #bbox.min[2] = z_center + (bbox.max[0] - x_center) * z_scale_back
    if 0:
        landmark_v = np.array([[bbox.min[0],bbox.max[1],bbox.max[2]],
                               [bbox.min[0], bbox.min[1], bbox.max[2]],
                               [bbox.max[0], bbox.max[1], bbox.max[2]],
                               [bbox.max[0], bbox.min[1], bbox.max[2]]
                               ])
        landmark_f =np.array([[0,1,2],[2,1,3]])

        landmark_v_color= np.zeros((landmark_v.shape[0],3))
        landmark_v_color[:,:] = [0,0,255]
        #merge_head_body_v, merge_head_body_v_f = add_vertex_faces(merge_head_body_v, merge_head_body_v_f, landmark_v, landmark_f)

        #merge_head_body_color = np.vstack((merge_head_body_color, landmark_v_color))
        for i in range(0,face_landmark_3d.shape[0]):
            plane_v = np.array([[-1, 1, bbox.max[2]],
                                [-1, -1, bbox.max[2]],
                                [1, 1, bbox.max[2]],
                                [1, -1, bbox.max[2]]
                                ])
            plane_v[:,0:2] *=0.01
            plane_v+=np.array([face_landmark_3d[i,0],face_landmark_3d[i,1],0])
            plane_f = np.array([[0, 1, 2], [2, 1, 3]])
            plane_v_color = np.zeros((plane_v.shape[0], 3))
            plane_v_color[:, :] = [0, 0, 255]
            merge_head_body_v, merge_head_body_v_f = add_vertex_faces(merge_head_body_v, merge_head_body_v_f, plane_v, plane_f)
            merge_head_body_color = np.vstack((merge_head_body_color, plane_v_color))


    bbox.max[0] = (bbox.max[0]-  x_center) * x_scale + x_center
    bbox.min[0] = (bbox.min[0] - x_center) * x_scale + x_center
    bbox.max[1] = (bbox.max[1] - y_center) * y_scale_up + y_center
    bbox.min[1] = (bbox.min[1] - y_center) * y_scale_down + y_center
    bbox.max[2] = (bbox.max[2] - z_center) * z_scale_front + z_center
    bbox.min[2] = (bbox.min[2] - z_center) * z_scale_back + z_center

    x_range = bbox.max[0] - bbox.min[0]
    y_range = bbox.max[1] - bbox.min[1]
    z_range = bbox.max[2] - bbox.min[2]
    image_width = 100
    image_height = int(image_width/x_range*y_range)

    #bbox.addvertex_array(body_v)

    b = FileFilt()
    b.FindFile(dirr=hair_file_dir)
    count = 0
    target = 1

    start_name = 'strands00357_1000'
    start_key = 0
    for k in b.fileList:
        if k == '':
            continue
        print k.split("/")[-1]
        filename_split = k.split("/")[-1].split(".")
        print filename_split
        if len(filename_split) > 1:
            #       print str(filename_split[-2])
            file_name = str(filename_split[-2])
        else:
            file_name = str(filename_split[0])
        #if file_name !='strands00358_1000' and start_key !=1:
        #    continue
        #if file_name != 'strands00024_1000':
        #    continue
        if file_name == 'strands00356_1000':
            continue
        start_key = 1
        hair_v, hair_v_f, hair_t_v, hair_t_f, hair_n_v, hair_n_f = read_igl_obj(
            hair_file_dir + '/' + file_name + '.obj')
        hair_v = hair_v[:, 0:3]

        hair_bbox = BBox3f()
        hair_bbox.addvertex_array(hair_v)
        z_range = hair_bbox.max[2] - hair_bbox.min[2]
        hair_bbox.max[2]+=z_range*0.1
        hair_bbox.min[2]-=z_range*0.1
        #感觉这样更准确
        bbox.max[2] = hair_bbox.max[2]
        bbox.min[2] = hair_bbox.min[2]

        if use_vertex_color:
            hair_vertex_color = readVertexColor(k)
        else:
            hair_vertex_color = np.zeros((hair_v.shape[0], 3))
            hair_vertex_color[:, :] = [255, 0, 0]
        # merge to one mesh
        merge_v, merge_v_f = add_vertex_faces(hair_v, hair_v_f, merge_head_body_v, merge_head_body_v_f)
        merge_t_v, merge_t_f = add_vertex_faces(hair_t_v, hair_t_f, merge_head_body_t_v, merge_head_body_t_f)
        merge_n_v, merge_n_f = add_vertex_faces(hair_n_v, hair_n_f, merge_head_body_n_v, merge_head_body_n_f)
        merge_color = np.vstack((hair_vertex_color, merge_head_body_color))
        mesh = MetroMesh()
        mesh.set_mesh(v=merge_v, vertex_color=merge_color, normal=merge_n_v,
                      vt=merge_t_v, face=merge_v_f, n_face=merge_n_f, t_face=merge_t_f)
        #Mesh_render_to_image_withmy_bbox(hair_file_dir + '/render_seg_norm/'+file_name+'.png', mesh, image_width, image_height,bbox)
        if 0:
            Mesh_render_to_image_withmy_bbox(hair_file_dir + output_name + file_name + '.png', mesh, image_width,
                                         image_height, bbox)
        write_full_obj(merge_v,merge_v_f,np.array([]),np.array([]),np.array([]),np.array([]),merge_color,hair_file_dir + output_name + file_name + '.obj')
        bbox_list = [float(bbox.min[0]),float(bbox.min[1]),float(bbox.min[2]),float(bbox.max[0]),float(bbox.max[1]),float(bbox.max[2])]
        merge_color = merge_color.astype(np.int32)
        if 0:
            result = meshlab_python.Mesh_render_to_image_withmy_bbox(
                hair_file_dir + output_name + file_name + '.png',
                merge_v.tolist(), merge_v_f.tolist(), [], [], [], [], merge_color.tolist(),
                int(image_width*5), int(image_height*5), bbox_list)


        #Mesh_render_to_image(hair_file_dir + '/render_seg/'+file_name+'.png', mesh, 256, 256)
        # write_simple_obj(v,f,prj_dir+'/convert_hair/'+file_name+'_'+str(stand_num)+'.obj')
        count += 1
def caculate_EMD_distance_1d(H1,H2):
    import sys
    sys.path.insert(0, "D:/mprojects/EmdL1_v3/EMD/x64/Release")
    import  EMD_PYTHON
    EMD_PYTHON.EMD_1D(H1, H2)
'''
输入是左上角为原点的
'''
def caculate_hair_seg_bin(image,center):
    seg_image = np.zeros( (image.shape[0],image.shape[1],image.shape[2]))
    image[:,:,:] = image[::-1,:,:] #以左下角为原点
    image[:, :, :] = image[:, :, ::-1] #转化为RGB
    center_x = center[0]
    center_y = image.shape[0]-1-center[1]
    center = np.array([center_x,center_y])
    theta_array = np.linspace(0.0, 2* np.math.pi, 32, False) #32
    theata_bin = np.zeros(len(theta_array),dtype=np.int)
    grid = theta_array[1] - theta_array[0]
    for i in range(0,image.shape[0]):
        for j in range(0,image.shape[1]):
            y= i
            x = j
            p = np.array([x,y])
            dir_vector = p- center
            if np.linalg.norm(dir_vector) < 4 :
                continue
            if image[i,j,0] < 2.0 and image[i,j,1] >250.0 and image[i,j,2]<2.0: #绿色
                continue
            if image[i,j,0] < 2.0 and image[i,j,1] <2.0 and image[i,j,2]<2.0: #黑色
                continue
            seg_image[i,j] = get_color_from2d_dir(dir_vector)
            rad = get_rad_from2d_dir(dir_vector)
            index = int(rad / grid)
            if index < 0:
                index = 0
            if index > theta_array.size -1:
                index = theta_array.size -1
            left_v = theta_array[index]
            if index+1 > theta_array.size -1:
                right_v = 2*math.pi
            else:
                right_v = theta_array[index+1]
            if not( rad >= left_v and rad <= right_v):
                print "bin index wrong"
            theata_bin[index]+=1

    seg_image[:,:,:] = seg_image[::-1,:,:]
    seg_image[:, :, :] = seg_image[:, :, ::-1]
    return seg_image,theata_bin
def caculate_hair_seg_bin_batch(input_seg,input_dir,out_seg_bin,out_dir_bin):
    import  cv2

    b = FileFilt()
    b.FindFile(dirr=input_seg)
    count = 0
    target = 1
    bin_map = {}
    for k in b.fileList:
        if k == '':
            continue
#       print k.split("/")[-1]
        #print k.split("/")
        filename_split = k.split("/")[-1].split(".")
#        print filename_split
        if len(filename_split) > 1:
            print str(filename_split[-2])
            file_name = str(filename_split[-2])
        else:
            file_name = str(filename_split[0])
        img = cv2.imread(k,cv2.IMREAD_COLOR )
        y_scale_up = 4.0
        y_scale_down = 20.0

        seg_img,theata_bin = caculate_hair_seg_bin(img,[(img.shape[1]-1)/2.0,(img.shape[0]-1)/(1+y_scale_down/y_scale_up)])
        img_dir = cv2.imread(input_dir+file_name+'.png', cv2.IMREAD_COLOR)
        #print  theata_bin
        bin_map[file_name] = theata_bin
        # cv2.imwrite(out_seg_bin+file_name
        #             + '.png', seg_img)
        #cv2.imwrite(out_dir_bin+file_name
        #            + '.png', seg_img)

    #print  bin_map
    save_binary_pickle(bin_map,'E:/workspace/dataset/hairstyles/hair/convert_hair_dir/seg_bin/seg_bin.pkl')
class EMD_DIS(object):
    def __init__(self,dis,name,flip,isuse_emd):
        self.dis = dis
        self.name = name
        self.flip = flip
        self.isuse_emd = isuse_emd
        pass
    def __cmp__(self,pe):
        if __lt__(pe) == True:
            return -1
        elif  __eq__(pe) == True:
            return 0
        else:
            return 1

    def __lt__(self, pe):
        if self.dis< pe.dis:
            return True
        else:
            return False

    def __eq__(self, pe):
        if self.dis == pe.dis:
            return True
    def __gt__(self,pe):
        if self.dis > pe.dis:
            return True
        else:
            return False

def seg_shape_similarity():
    hairs_seg_bin = load_binary_pickle('E:/workspace/dataset/hairstyles/hair/convert_hair_dir/seg_bin/seg_bin.pkl')
    import  sys
    sys.path.insert(0, "E:/workspace/EmdL1_v3/EMD/x64/Release/")
    from EMD_PYTHON import EMD_1D

    source = 'strands00001_1000'
    source_bin = hairs_seg_bin[source]
    emd_array =[]
    use_emd = True

    print EMD_1D([1,2,3,5], [2,4,6,10])
    print EMD_1D([1, 2, 3, 5], [4, 8, 12, 20])
    for key, value in hairs_seg_bin.items():
        #print key, 'corresponds to', value
        if use_emd:
            distance1 = EMD_1D(source_bin.tolist(), value.tolist())
        else:
            distance1 = np.linalg.norm(source_bin-value)
        emd1 = EMD_DIS(distance1,key,False,use_emd)
        emd_array.append(emd1)
        if use_emd:
            distance2 = EMD_1D(source_bin.tolist(), value[::-1].tolist())
        else:
            distance2 = np.linalg.norm(source_bin - value[::-1])
        emd2 = EMD_DIS(distance2, key, True,use_emd)
        emd_array.append(emd2)
    emd_array.sort()
    safe_mkdirs('E:/workspace/dataset/hairstyles/hair/convert_hair_dir/seg_bin/'+source+'_similarity'+str(use_emd)+'/')
    import  cv2
    for i in range(0,len(emd_array)):
        img_path = 'E:/workspace/dataset/hairstyles/hair/convert_hair_dir/seg_bin/'+emd_array[i].name+'.png'
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if emd_array[i].flip:
            img = img[:,::-1,:]
        cv2.imwrite('E:/workspace/dataset/hairstyles/hair/convert_hair_dir/seg_bin/'+source+'_similarity'+str(use_emd)+'/'
                    +str(i)+'.png'
                    ,img)
        #print emd_array[i].dis,emd_array[i].name,emd_array[i].flip
def seg_dir_similarity():

    hair_seg_dir_path = 'E:/workspace/dataset/hairstyles/hair/convert_hair_dir/render_hair_dir_seg/'
    import cv2
    source = 'strands00153_1000'
    img_source = cv2.imread(hair_seg_dir_path + source + '.png', cv2.IMREAD_COLOR)
    emd_array = []
    use_emd = 0
    b = FileFilt()
    b.FindFile(dirr=hair_seg_dir_path)
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
        else:
            file_name = str(filename_split[0])
        img_seg = cv2.imread(k, cv2.IMREAD_COLOR)
        if use_emd:
            pass
        else:
            dis = dir_seg_distance(img_source, img_seg)
        emd1 = EMD_DIS(dis,file_name,False)
        emd_array.append(emd1)
        if use_emd:
            pass
        else:
            dis = dir_seg_distance(img_source, img_seg[:,::-1,:])
        emd1 = EMD_DIS(dis,file_name,True)
        emd_array.append(emd1)
    emd_array.sort()
    safe_mkdirs('E:/workspace/dataset/hairstyles/hair/convert_hair_dir/render_hair_dir_seg/' + source + '_similarity' + str(
            use_emd) + '/')
    for i in range(0,len(emd_array)):
        print emd_array[i].name,emd_array[i].dis,emd_array[i].flip
        img_path = 'E:/workspace/dataset/hairstyles/hair/convert_hair_dir/render_hair_dir_seg/'+emd_array[i].name+'.png'
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if emd_array[i].flip:
            img = img[:,::-1,:]
        cv2.imwrite('E:/workspace/dataset/hairstyles/hair/convert_hair_dir/render_hair_dir_seg/'+source+'_similarity'+str(use_emd)+'/'
                    +str(i)+'.png',img)
@jit(nopython =True,cache=True)
def dir_seg_distance(seg_img1,seg_img2):
    if (seg_img1.shape[0] != seg_img2.shape[0] or seg_img1.shape[1] != seg_img2.shape[1]):
        print 'seg_img shape != dir_img shape'

    seg_img_rad1 = np.zeros(seg_img1.shape)
    seg_img_rad2 = np.zeros(seg_img2.shape)
    common_area = np.zeros(seg_img2.shape) #, np.bool
    distance=0
    count = 0 #
    for i in range(0,seg_img1.shape[0]):
        for j in range(0, seg_img1.shape[1]):

            if (seg_img1[i, j, 0] < 2.0 and seg_img1[i, j, 1] <2.0 and seg_img1[i, j, 2] < 2.0) \
                    or (seg_img2[i, j, 0] < 2.0 and seg_img2[i, j, 1] <2.0 and seg_img2[i, j, 2] < 2.0):
                common_area[i,j] = False
                continue
            else:
                #只计算共同区域
                if 0:
                    seg_img_rad1[i, j, :] = get_rad_from_color(seg_img1[i, j, :])
                    seg_img_rad2[i, j, :] = get_rad_from_color(seg_img2[i, j, :])
                    dis = np.linalg.norm(seg_img_rad1[i, j, :]- seg_img_rad2[i, j, :])
                    distance += dis
                    common_area[i, j] = True
                    count+=1
                else:
                    dis = np.linalg.norm(seg_img1[i, j, :]- seg_img2[i, j, :])
                    distance += dis
                    common_area[i, j] = True
                    count+=1
    #print distance, count,distance/count
    return  distance/count #使用平均距离


def generate_hair_dir_single():
    hair_seg_path = 'E:/workspace/dataset/hairstyles/hair/convert_hair_dir/render_hair_seg_body/'
    hair_dir_path = 'E:/workspace/dataset/hairstyles/hair/convert_hair_dir/render_hair_dir_body/'
    hair_seg_dir_path = 'E:/workspace/dataset/hairstyles/hair/convert_hair_dir/render_hair_dir_seg/'
    import cv2
    b = FileFilt()
    b.FindFile(dirr=hair_seg_path)

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
        else:
            file_name = str(filename_split[0])
        img_seg = cv2.imread(k, cv2.IMREAD_COLOR)
        img_dir = cv2.imread(hair_dir_path+file_name+'.png', cv2.IMREAD_COLOR)
        img_dir_seg = img_dir.copy()
        for i in range(0, img_dir_seg.shape[0]):
            for j in range(0, img_dir_seg.shape[1]):
                if img_seg[i, j, 0] < 2.0 and img_seg[i, j, 1] > 250.0 and img_seg[i, j, 2] < 2.0:  # 绿色
                    img_dir_seg[i, j, :] = [0,0,0]

        cv2.imwrite(hair_seg_dir_path+file_name+'.png',img_dir_seg)

def test():
    theta_array = np.linspace(0.0, 2 * np.math.pi, 32, False)  # 32
    for i in range(0,len(theta_array)):
        if i == len(theta_array)-1:
            pass
        rad = theta_array[i]
        color = get_color_from_rad(rad)
        new_rad = get_rad_from_color(color)
        print  rad,color,new_rad

def test_scale_bbox():
    import cv2
    from triangle_raster import BBoxi_2d
    from fitting.util import  rescale_imge_with_bbox,read_landmark
    input_dir = 'E:/workspace/dataset/hairstyles/2d_hair/Seg_refined/'
    out_put_dir = 'E:/workspace/dataset/hairstyles/2d_hair/rescale/'
    landmark_file = 'E:/workspace/vrn_data/bgBlue/A1301043678290A/2d/A1301043678290A.txt'
    ori_img = cv2.imread(input_dir+'A1301043678290A.png',cv2.IMREAD_COLOR)
    height = ori_img.shape[0]
    width = ori_img.shape[1]
    landmark_2d = read_landmark(landmark_file)
    landmark_2d[:, 1] = height - 1 - landmark_2d[:, 1]  # 坐标系原点转化为左下角
    landmark_2d = landmark_2d[17:,:]
    #我们计算它的包围盒

    bbox = BBoxi_2d(landmark_2d)
    lanmark_crop_img = rescale_imge_with_bbox(ori_img[::-1,:,:],bbox)
    cv2.imwrite(out_put_dir + 'A1301043678290A_landmark_crop.png', lanmark_crop_img[::-1, :, :])

    x_center = (bbox.max[0] + bbox.min[0])/2
    y_center = (bbox.max[1] + bbox.min[1]) / 2
    x_scale =5
    y_scale_up = 4.0
    y_scale_down = 20.0

    #利用x 来估计z
    #bbox.max[2] = z_center +  (bbox.max[0]-  x_center)*z_scale_front
    #bbox.min[2] = z_center + (bbox.max[0] - x_center) * z_scale_back

    bbox.max[0] = int((bbox.max[0]-  x_center) * x_scale + x_center)
    bbox.min[0] = int((bbox.min[0] - x_center) * x_scale + x_center)
    bbox.max[1] =  int((bbox.max[1] - y_center) * y_scale_up + y_center)
    bbox.min[1] =  int((bbox.min[1] - y_center) * y_scale_down + y_center)

    x_range = bbox.max[0] - bbox.min[0]
    y_range = bbox.max[1] - bbox.min[1]

    image_width = 100
    image_height = 213 #int(image_width/x_range*y_range)

    # bbox_2d = BBoxi()
    # bbox_2d.min[0] = -1
    # bbox_2d.min[1] = -1
    # bbox_2d.max[0] = width
    # bbox_2d.max[1] = height
    scaled_img = rescale_imge_with_bbox(ori_img[::-1,:,:],bbox)
    cv2.imwrite(out_put_dir + 'A1301043678290A_1.png', scaled_img[::-1, :, :])
    scaled_img = cv2.resize(scaled_img, (image_width, image_height), interpolation=cv2.INTER_NEAREST)  #cv2.INTER_CUBIC
    for j in range(0,scaled_img.shape[0]):
        for i in range(0,scaled_img.shape[1]):
            if scaled_img[j,i,2] < 254:
                scaled_img[j, i, :] = [0,0,0]
    cv2.imwrite(out_put_dir+'A1301043678290A_2.png',scaled_img[::-1,:,:])


def get_similar_hair_from_database_wraper(input_seg_img,input_dir_img,seg_database_path,seg_database_input_dir,direction_database_dir,out_put_dir):
    hair_seg_dir_path = 'E:/workspace/dataset/hairstyles/hair/convert_hair_dir/render_hair_dir_seg/'
    bin_map = {}
    img = input_seg_img
    y_scale_up = 4.0
    y_scale_down = 20.0

    seg_img, theata_bin = caculate_hair_seg_bin(img, [(img.shape[1] - 1) / 2.0,
                                                      (img.shape[0] - 1) / (1 + y_scale_down / y_scale_up)])

    # print  theata_bin
    source_bin = theata_bin
    hairs_seg_bin = load_binary_pickle(seg_database_path) #load_binary_pickle('E:/workspace/dataset/hairstyles/hair/convert_hair_dir/seg_bin/seg_bin.pkl')
    import  sys
    sys.path.insert(0, "E:/workspace/EmdL1_v3/EMD/x64/Release/")
    from EMD_PYTHON import EMD_1D

    emd_array =[]
    use_emd = 0

    # print EMD_1D([1,2,3,5], [2,4,6,10])
    # print EMD_1D([1, 2, 3, 5], [4, 8, 12, 20])
    img_source = input_dir_img
    img_source = img_source.astype(np.float)
    import cv2
    weight_c = 0.5
    for key, value in hairs_seg_bin.items():
        #print key, 'corresponds to', value
        if use_emd:
            distance1 = EMD_1D(source_bin.tolist(), value.tolist())
        else:
            distance1 = np.linalg.norm(source_bin-value)
        k=hair_seg_dir_path+key+'.png'
        img_seg = cv2.imread(k, cv2.IMREAD_COLOR)
        img_seg = img_seg.astype(np.float)
        dis1 = dir_seg_distance(img_source, img_seg)
        print  distance1 ,dis1
        emd1 = EMD_DIS(distance1+weight_c*dis1,key,False,use_emd)
        emd_array.append(emd1)
        if use_emd:
            distance2 = EMD_1D(source_bin.tolist(), value[::-1].tolist())
        else:
            distance2 = np.linalg.norm(source_bin - value[::-1])
        dis2 = dir_seg_distance(img_source, img_seg[:, ::-1, :])
        print  distance2, dis2
        emd2 = EMD_DIS(distance2+weight_c*dis2, key, True,use_emd)
        emd_array.append(emd2)
    emd_array.sort()
    safe_mkdirs(out_put_dir+'/')

    #计算方向间距离

    if 0:
        hair_seg_dir_path = 'E:/workspace/dataset/hairstyles/hair/convert_hair_dir/render_hair_dir_seg/'
        import cv2
        source = 'source_'
        img_source = input_dir_img #cv2.imread(hair_seg_dir_path + source + '.png', cv2.IMREAD_COLOR)
        img_source = img_source.astype(np.float)

        emd_array_dir = []
        use_emd = 0
        b = FileFilt()
        b.FindFile(dirr=hair_seg_dir_path)
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
            else:
                file_name = str(filename_split[0])
            img_seg = cv2.imread(k, cv2.IMREAD_COLOR)
            img_seg = img_seg.astype(np.float)
            if use_emd:
                pass
            else:
                dis = dir_seg_distance(img_source, img_seg)
            emd1 = EMD_DIS(dis, file_name,False, use_emd)
            emd_array_dir.append(emd1)
            if use_emd:
                pass
            else:
                dis = dir_seg_distance(img_source, img_seg[:, ::-1, :])
            emd1 = EMD_DIS(dis, file_name,True, use_emd)
            emd_array_dir.append(emd1)
        emd_array_dir.sort()
        safe_mkdirs(
            out_put_dir + source + '_similarity' + str(
                use_emd) + '/')
        for i in range(0, len(emd_array_dir)):
            print emd_array_dir[i].name, emd_array_dir[i].dis, emd_array_dir[i].flip
            img_path = 'E:/workspace/dataset/hairstyles/hair/convert_hair_dir/render_hair_dir_seg/' + emd_array_dir[
                i].name + '.png'
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if emd_array_dir[i].flip:
                img = img[:, ::-1, :]
            cv2.imwrite(
                out_put_dir + source + '_similarity' + str(
                    use_emd) + '/'
                + str(i).zfill(3) +'_'+ emd_array_dir[i].name+'_flip_'+str(emd_array_dir[i].flip)+'.png', img)

    import  cv2

    # cv2.imwrite(out_put_dir + '/' + 'soure_seg_bin' + '.png',
    #             seg_img)
    #从shape 排序中筛选出方向最相似的
    img_source = input_dir_img  # cv2.imread(hair_seg_dir_path + source + '.png', cv2.IMREAD_COLOR)
    img_source = img_source.astype(np.float)
    min_dis = 100000
    min_dis_corr_idx = -1
    for i in range(0, len(emd_array)):
        if i > 10: #支取最多10个
            break
        pass
        direction_img_path = direction_database_dir + '/' + emd_array[i].name + '.png'
        diretion_img = cv2.imread(direction_img_path, cv2.IMREAD_COLOR)
        if emd_array[i].flip:
            diretion_img = diretion_img[:, ::-1, :]
            diretion_img = diretion_img.astype(np.float)

            dis = dir_seg_distance(img_source, diretion_img)
            if dis < min_dis :
                min_dis_corr_idx = i
    print min_dis_corr_idx
    direction_img_path = direction_database_dir + '/' + emd_array[min_dis_corr_idx].name + '.png'
    diretion_img = cv2.imread(direction_img_path, cv2.IMREAD_COLOR)
    safe_mkdirs(
        out_put_dir + 'best' + str(
            use_emd) + '/')
    if emd_array[min_dis_corr_idx].flip:
        diretion_img = diretion_img[:, ::-1, :]
    cv2.imwrite(out_put_dir + 'best' + str(
        use_emd)  + '/' + 'dir_'+'useemd_' + str(use_emd) + '_' + str(min_dis_corr_idx) + '_corr_' + emd_array[min_dis_corr_idx].name + '_flip_' + str(
            emd_array[min_dis_corr_idx].flip) + '.png', diretion_img)

    best_array = emd_array[min_dis_corr_idx]

    emd_array.insert(0,best_array)
    safe_mkdirs(
        out_put_dir + 'shape_similarity' + str(
            use_emd) + '/')
    for i in range(0,len(emd_array)):
        if i > 20: #支取最多10个
            break
        seg_bin_img_path =seg_database_input_dir+'/'+emd_array[i].name+'.png'
        direction_img_path = direction_database_dir+'/'+emd_array[i].name+'.png'
        seg_bin_img = cv2.imread(seg_bin_img_path, cv2.IMREAD_COLOR)
        diretion_img = cv2.imread(direction_img_path, cv2.IMREAD_COLOR)
        if emd_array[i].flip:
            seg_bin_img = seg_bin_img[:,::-1,:]
            diretion_img= diretion_img[:,::-1,:]
        #cv2.imwrite(out_put_dir+'/' +'useemd_'+str(use_emd)+'_'+str(i)+'_corr_'+emd_array[i].name+'_flip_'+str(emd_array[i].flip)+'.png' ,seg_bin_img)
        cv2.imwrite(out_put_dir + 'shape_similarity' + str(
            use_emd)  + '/' + 'dir_'+'useemd_' + str(use_emd) + '_' + str(i) + '_corr_' + emd_array[i].name + '_flip_' + str(
                emd_array[i].flip) + '.png', diretion_img)
    return emd_array #emd_array_dir #emd_array

def caculate_lanmark_bbox(landmark_2d):
    from triangle_raster import BBoxi_2d
    bbox = BBoxi_2d(landmark_2d)


    x_center = (bbox.max[0] + bbox.min[0])/2
    y_center = (bbox.max[1] + bbox.min[1]) / 2
    x_scale =5*0.8
    y_scale_up = 4.0*0.8
    y_scale_down = 20.0*0.8

    #利用x 来估计z
    #bbox.max[2] = z_center +  (bbox.max[0]-  x_center)*z_scale_front
    #bbox.min[2] = z_center + (bbox.max[0] - x_center) * z_scale_back

    bbox.max[0] = int((bbox.max[0]-  x_center) * x_scale + x_center)
    bbox.min[0] = int((bbox.min[0] - x_center) * x_scale + x_center)
    bbox.max[1] =  int((bbox.max[1] - y_center) * y_scale_up + y_center)
    bbox.min[1] =  int((bbox.min[1] - y_center) * y_scale_down + y_center)


    return bbox

def get_similar_hair_from_database(input_seg_img,input_dir_img,out_put_dir):
    seg_database_path = 'E:/workspace/dataset/hairstyles/hair/convert_hair_dir/seg_bin/seg_bin.pkl'
    seg_database_input_dir = 'E:/workspace/dataset/hairstyles/hair/convert_hair_dir/seg_bin/'
    direction_database_dir = 'E:/workspace/dataset/hairstyles/hair/convert_hair_dir/render_hair_dir_seg/'
    emd_array = get_similar_hair_from_database_wraper(
        input_seg_img = input_seg_img, input_dir_img=input_dir_img, seg_database_path=seg_database_path,
        seg_database_input_dir=seg_database_input_dir,direction_database_dir=direction_database_dir, out_put_dir=out_put_dir)
    return  emd_array

def build_hair_for_img_simgle(object_name,input_ori_img_file ,input_seg_img_file,input_dir_img_file,input_landmark_file,out_put_dir,project_dir):
    import cv2
    from triangle_raster import BBoxi_2d
    from fitting.util import  rescale_imge_with_bbox,read_landmark
    safe_mkdirs(out_put_dir + '/')
    #landmark_file = 'E:/workspace/vrn_data/bgBlue/A1301043678290A/2d/A1301043678290A.txt'
    ori_img = cv2.imread(input_ori_img_file, cv2.IMREAD_COLOR)
    seg_img = cv2.imread(input_seg_img_file, cv2.IMREAD_COLOR)
    dir_img = cv2.imread(input_dir_img_file, cv2.IMREAD_COLOR)
    height = ori_img.shape[0]
    width = ori_img.shape[1]
    landmark_2d = read_landmark(input_landmark_file)
    landmark_2d[:, 1] = height - 1 - landmark_2d[:, 1]  # 坐标系原点转化为左下角
    selected_landmark_2d = landmark_2d[17:, :]
    from triangle_raster import BBoxi_2d
    bbox = BBoxi_2d(selected_landmark_2d)
    bbox_list = [bbox.min[0],bbox.min[1],bbox.max[0],bbox.max[1]]
    lanmark_crop_img = rescale_imge_with_bbox(ori_img[::-1,:,:],bbox_list)
    cv2.imwrite(out_put_dir + '2d_landmark_crop.png', lanmark_crop_img[::-1, :, :])
    bbox = caculate_lanmark_bbox(selected_landmark_2d)
    x_range = bbox.max[0] - bbox.min[0]
    y_range = bbox.max[1] - bbox.min[1]
    image_width = 100
    image_height = 213
    bbox_list = [bbox.min[0], bbox.min[1], bbox.max[0], bbox.max[1]]
    scaled_img = rescale_imge_with_bbox(seg_img[::-1,:,:],bbox_list)
    scaled_dir_img = rescale_imge_with_bbox(dir_img[::-1,:,:],bbox_list)
    #cv2.imwrite(out_put_dir +'A1301043678290A_1.png', scaled_img[::-1, :, :])
    scaled_img = cv2.resize(scaled_img, (image_width, image_height), interpolation=cv2.INTER_NEAREST)  #cv2.INTER_CUBIC
    scaled_dir_img = cv2.resize(scaled_dir_img, (image_width, image_height), interpolation=cv2.INTER_NEAREST)
    for j in range(0,scaled_img.shape[0]):
        for i in range(0,scaled_img.shape[1]):
            if scaled_img[j,i,2] < 254:
                scaled_img[j, i, :] = [0,0,0]
            if scaled_dir_img[j,i,0] ==255 and scaled_dir_img[j,i,1]==255 and scaled_dir_img[j,i,2]==255:
                scaled_dir_img[j, i, :]=[0,0,0]
                pass
    cv2.imwrite(out_put_dir+'/'+object_name+'_scaled.png',scaled_img[::-1,:,:])
    cv2.imwrite(out_put_dir + '/' + object_name + '_dir_scaled.png', scaled_dir_img[::-1, :, :])
    input_seg_img = cv2.imread(out_put_dir+'/'+object_name+'_scaled.png',cv2.IMREAD_COLOR)
    input_dir_img = cv2.imread(out_put_dir+'/'+object_name+'_dir_scaled.png',cv2.IMREAD_COLOR)
    #out_put_dir =''
    emd_array = get_similar_hair_from_database(input_seg_img, input_dir_img, out_put_dir)
    #根据landmark ，以及Frame 模型3d landmark点 得到 选择矩阵，放缩矩阵，平移
    # R =[]
    # Scale =[]
    # T =[]
    #根据 hiar 中对应的头部模型变换到 frame 初始坐标的变换，把hair 变换到该坐标
    # R_Local =[]
    # Scale_local = 1
    # T_local =[]
    from fitting.util import get_opt_transform_3d,get_opt_transform_2d,readImage,sample_color_from_img
    hair_mesh_obj_database_dir = 'E:/workspace/dataset/hairstyles/hair/convert_hair_dir/'
    model_dir = 'E:/workspace/dataset/hairstyles/'
    v_frame_aligned, f_frame_aligned, t_frame_aligned, t_f_frame_aligned, n_frame_aligned, n_f_frame_aligned = read_igl_obj(model_dir+'frame_aligned.obj')
    v_frame_init, f_frame_init, t_frame_init, t_f_frame_init, n_frame_init, n_f_frame_init = read_igl_obj(model_dir +'frame_female_init.obj')
    # landmark embedding
    lmk_emb_path = './data/lmk_embedding_intraface_to_flame.pkl'
    from fitting.util import mesh_points_by_barycentric_coordinates
    from fitting.landmarks import load_embedding
    lmk_face_idx, lmk_b_coords = load_embedding(lmk_emb_path)
    from  fitting.util import mesh_points_by_barycentric_coordinates,get_vertex_normal,sample_color_from_img
    frame_landmark_3d = mesh_points_by_barycentric_coordinates(v_frame_init, f_frame_init, lmk_face_idx, lmk_b_coords)
    landmark_idx = range(17, 60) + range(61, 64) + range(65, 68)
    cast_landmark = landmark_2d[landmark_idx,:]
    if 0:
        Scale, R, T = get_opt_transform_2d(frame_landmark_3d, cast_landmark)
    else:
        from face_generate import  generate_face
        import  os
        if os.path.exists(project_dir+object_name+'/'+'generate_face/' + '/face_result.pkl'):
            pass

            result = load_binary_pickle(filepath=project_dir+object_name+'/'+'generate_face/' + '/face_result.pkl')
            frame_re_texture_map = 'D:\mproject/face-reconstruct/texpc\source_para/texture/frame_template_retex.obj'
            v_frame_re_texture, f_frame_re_texture, t_frame_re_texture, t_f_frame_re_texture, n_re_texture, n_f_re_texture = read_igl_obj(
                frame_re_texture_map)
            v_aligned = result['mesh_v']
            face_aligned_f = result['mesh_f']
            oriimage_file_path =project_dir +object_name+'/' + object_name + '.jpg'
            ori_image = readImage(oriimage_file_path)
            ori_image = ori_image[::-1, :, :]
            from fitting.util import mesh_loop,get_vertex_normal
            v_noraml_aligned = get_vertex_normal(v_aligned,face_aligned_f)
            mesh_loop(v_aligned, face_aligned_f, v_noraml_aligned, face_aligned_f, t_frame_re_texture,
                      t_f_frame_re_texture, np.array([]),
                      project_dir+object_name+'/'+'generate_face/'+'generate_face_subdiv.obj', 2)
            v_subdiv, f_subdiv, t_subdiv, t_f_subdiv, n_subdiv, n_f_subdiv = read_igl_obj(
                project_dir + object_name + '/' + 'generate_face/' + 'generate_face_subdiv.obj')
            invaid_img = np.zeros((seg_img.shape[0],seg_img.shape[1]),np.bool)
            for j in range(0,seg_img.shape[0]):
                for i in range(0,seg_img.shape[1]):
                    if seg_img[j,i,0] <1 and seg_img[j,i,1] >254 and seg_img[j,i,2]<1:
                        invaid_img[j,i] = True
            invaid_img = invaid_img[::-1,:]
            v_color = sample_color_from_img(v_subdiv, n_subdiv, ori_image,invaid_img= invaid_img)
            write_full_obj(v_subdiv, f_subdiv, n_subdiv, n_f_subdiv, t_subdiv, t_f_subdiv, v_color,
                           out_put_dir + '/face_with_subiv_texture.obj')

        else:
            result =generate_face(vrn_object_dir=project_dir+object_name+'/',
                                  object_name=object_name,
                                  project_dir =project_dir,
                                  frame_model_path='./models/female_model.pkl',
                          out_put_dir=project_dir+object_name+'/'+'generate_face/')
        v_aligned = result['mesh_v']
        face_aligned_f = result['mesh_f']
        parms = result['parms']
        T = parms['trans']
        pose =parms['pose']
        betas =parms['betas']
        R = parms['global_rotate']
        Scale = parms['scale']
        print T,R,Scale
        # model_path = './models/female_model.pkl'
        # from smpl_webuser.serialization import load_model
        # model = load_model(model_path)
        # model.betas[:] = betas
        # model.pose[:] = pose
        # Scale*model.r +T
        pass

    Scale = Scale[0]
    T = np.array([T[0],T[1],0])
    Scale_local,R_Local, T_local =get_opt_transform_3d(v_frame_aligned,v_frame_init)
    Scale_local = Scale_local[0]
    if 0:
        v_frame_align_to_image = (Scale*np.dot(R,v_frame_init.T)).T +T
    else:
        v_frame_align_to_image = v_aligned
    v_frame_align_to_init = (Scale_local * np.dot(R_Local , v_frame_aligned.T)).T + T_local
    #计算点法向
    vn_frame_align_to_image =get_vertex_normal(v_frame_align_to_image,f_frame_init)
    if 1:
        v_color_caculate = np.zeros(v_frame_align_to_image.shape,np.uint8)
        v_color_caculate[:,:] =[255,255,255]
        ray_dir = np.array([0,0,1])
        ray_dir = np.reshape(ray_dir,(3,1))
        dot_coef = np.dot(vn_frame_align_to_image, ray_dir)
        dot_coef = np.array([ 0 if x <0.01 else x for x in dot_coef ])
        dot_coef = np.reshape(dot_coef,(dot_coef.shape[0],1))
        v_color_caculate = v_color_caculate*dot_coef
        v_color_caculate = v_color_caculate*0.7+(1-0.7)*np.array([255,255,255])

    else:
        v_color_caculate = sample_color_from_img(v_frame_align_to_image,vn_frame_align_to_image,ori_img[::-1,:,:])
    v_color_caculate = v_color_caculate.astype(np.intc)
    write_full_obj(v_frame_align_to_image, f_frame_init,vn_frame_align_to_image, f_frame_init,t_frame_init, t_f_frame_init, v_color_caculate, out_put_dir + '/' +
    'frame_aligned_to_image' + '.obj')
    write_full_obj(v_frame_align_to_init, f_frame_aligned, n_frame_aligned, n_f_frame_aligned,t_frame_aligned, t_f_frame_aligned,  np.array([]), out_put_dir + '/' +
                   'frame_aligned_to_init' + '.obj')
    write_landmark_to_obj(out_put_dir + '/' +
                   '2d_landmark' + '.obj',np.hstack((landmark_2d,np.zeros((landmark_2d.shape[0],1)))))

    frame_landmark_3d = (Scale * np.dot(R, frame_landmark_3d.T)).T + T
    write_landmark_to_obj(out_put_dir + '/' +
                   'frame_landmark_3d' + '.obj',frame_landmark_3d)
    for i in range(0,len(emd_array)):
        if i > 20: #支取最多10个
            break
        dis = emd_array[i].dis
        strand_name = emd_array[i].name
        flip = emd_array[i].flip
        isuse_emd = emd_array[i].isuse_emd
        corr_mesh_file = hair_mesh_obj_database_dir +'/'+strand_name+'.obj'
        v_hair, f_hair, t, t_f, n, n_f = read_igl_obj(corr_mesh_file)
        v_color = readVertexColor(corr_mesh_file)
        if flip:
            v_hair[:,0] =-v_hair[:,0]
            f_hair = f_hair[:,::-1]
        v_hair_transformed = Scale*np.dot(R,( (Scale_local*np.dot(R_Local,v_hair.T)).T+T_local).T).T+T   #+np.array([0,-30,10])
        if i<30:
            write_full_obj(v_hair_transformed, f_hair, n, n_f,t, t_f,v_color, out_put_dir+'/'+
                           'useemd_' + str(isuse_emd) + '_' + str(i) + '_corr_' + strand_name + '_flip_' + str(flip)+'.obj')
        if 1:
            sys.path.insert(0, "D:/mproject/meshlab2016/meshlab/src/x64/Release/")
            import meshlab_python
            bbox_list = [float(0), float(0), float(-600), float(width),
                         float(height), float(600)]
            v_color = v_color.astype(np.int32)
            merge_v,merge_f = add_vertex_faces(v_hair_transformed,f_hair,v_frame_align_to_image, f_frame_init)
            merge_color = np.vstack((v_color,v_color_caculate))
            if 0:
                result = meshlab_python.Mesh_render_to_image_withmy_bbox( out_put_dir+'/'+
                    'useemd_' + str(isuse_emd) + '_' + str(i) + '_corr_' + strand_name + '_flip_' + str(flip) +'render_to_image'+ '.png',
                    v_hair_transformed.tolist(), f_hair.tolist(), [], [], [], [], v_color.tolist(),
                    int(width), int(height), bbox_list)
            else:
                result = meshlab_python.Mesh_render_to_image_withmy_bbox( out_put_dir+'/'+
                    'useemd_' + str(isuse_emd) + '_' + str(i) + '_corr_' + strand_name + '_flip_' + str(flip) +'render_to_image'+ '.png',
                    merge_v.tolist(), merge_f.tolist(), [], [], [], [], merge_color.tolist(),
                    int(width), int(height), bbox_list)
                render_img = np.array(result[0])
                z_buffer_img = np.array(result[1])
                cv2.imwrite(out_put_dir+'/'+'render_img.png',render_img[:,:,::-1])
                cv2.imwrite(out_put_dir + '/' + 'z_buffer_img.png',z_buffer_img)
                #print result


def build_hair_for_img_batch():

    # img_dir = 'E:/workspace/dataset/hairstyles/2d_hair/'
    # landmark_dir = 'E:/workspace/vrn_data/bgBlue/'
    img_dir = 'E:\workspace/vrn_data\hair_crop\man_crop/'
    landmark_dir = 'E:\workspace/vrn_data\hair_crop\man_crop/Landmark/'
    # img_dir = 'E:\workspace/vrn_data/bgBlue/'
    # landmark_dir = 'E:\workspace/vrn_data/bgBlue\Landmark/'
    # img_dir = 'E:\workspace/vrn_data\hair_crop\girl_crop/'
    # landmark_dir = 'E:\workspace/vrn_data\hair_crop\girl_crop/Landmark/'
    import cv2
    b = FileFilt()
    b.FindFile(dirr=img_dir)
    skip_file =['hairgirl_000','hairgirl_001','hairman_002','hairman_003','hairman_005','hairman_013','hairman_023','hairman_024']
    processfile = ['hairgirl_000','hairgirl_001','hairman_002','hairman_003']
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
            fomat_name =  str(filename_split[-1])
        else:
            file_name = str(filename_split[0])
            fomat_name =''
        # if 'A13010436665609' != file_name:
        #      continue
        if 'directed_color' == file_name or 'undirected_color' == file_name:
            continue
        # if file_name in skip_file:
        #     continue
        if file_name in processfile:
            pass
        else:
            continue
        build_hair_for_img_simgle(object_name = file_name,input_ori_img_file=img_dir+'/'+file_name+'.'+fomat_name, input_seg_img_file = img_dir+'Seg_refined/'+file_name+'.png',
                                    input_dir_img_file = img_dir + 'Strand/' + file_name + '.png',
                       #           input_landmark_file = landmark_dir+file_name+'/'+'2d/'+file_name+'.txt',
                                  input_landmark_file=landmark_dir + file_name +  '.txt',
                                  out_put_dir=img_dir+'result/'+file_name+'/'+'builded_hair/',
                                  project_dir = img_dir)


def continuous_str(lst):
    j = 0
    str1 = ''
    for i, item in enumerate(lst):
        if i > 0:
            if lst[i] != lst[i - 1] + 1:
                tmp = lst[j:i]
                if len(tmp) == 1:
                    str1 += str(tmp[0]) + ','
                else:
                    str1 += str(tmp[0]) + "~" + str(tmp[-1]) + ','
                j = i
    tmp2 = lst[j:]
    if len(tmp2) == 1:
        str1 += str(tmp2[0]) + ','
    else:
        str1 += str(tmp2[0]) + "~" + str(tmp2[-1]) + ','

    return str1[:-1]
'''
alpha , 从负z轴开始，绕y顺时针旋转的弧度
belta ,与位于原点的投影平面的夹角（弧度) ，平面上为负
dis , 到原点的距离
'''
def caculate_single_strand_dis(strand ,max_belta = pi/10,max_alpha = pi/180):
    max_dis = -1
    corr_alpha =-1
    corr_belta = -1
    corr_idx = 0
    epslion = 0.0001
    if len(strand) == 1:
        #print 'len(strand)',len(strand)
        return corr_alpha,corr_belta,max_dis
    for i in range(0, len(strand)): #最后一个点的tangent 来源于前一个
        point_pos = strand[i][:]
        x = point_pos[0]
        y = point_pos[1]
        z = point_pos[2]
        point_dir = np.array([x,y,z])
        plane_proj_dir =np.array([x,0,z])
        if np.linalg.norm(point_dir) < epslion:
            continue
        if np.linalg.norm(plane_proj_dir) < epslion:
            continue
        point_dir/= np.linalg.norm(point_dir)
        plane_proj_dir/=np.linalg.norm(plane_proj_dir)
        dot_result = np.dot(point_dir,plane_proj_dir)
        belta = math.acos(dot_result)
        if abs(belta) > max_belta: #剔除大于max_belta的采样点
            continue
        if z <0 :
            belta = -belta
        dir = [x,-z]
        alpha = get_rad_from2d_dir(dir)
        if alpha <0:
            print alpha,dir
        dis = np.linalg.norm(point_pos)
        if dis > max_dis:
            max_dis = dis
            corr_alpha = alpha
            corr_belta = belta
            corr_idx = i

    return  corr_alpha,corr_belta ,max_dis
def split_connected_list(in_list):
    connected_list =[]
    connected_list.append([])
    connected_list[0].append(in_list[0])
    for i in range(1,len(in_list)):
        if in_list[i] == in_list[i-1]+1:
            connected_list[len(connected_list)-1].append(in_list[i])
        else:
            connected_list.append([])
            connected_list[len(connected_list) - 1].append(in_list[i])
    return connected_list

def strands_key_idx(input_strands,horizontal_grid=100,vertical_range = pi/90):
    #horizontal_grid = 100 # 水平面顺时针分成100圈
    #vertical_range = pi/90 #垂直的最大弧度
    horizontal_grid_range = 2*pi /horizontal_grid
    num = 10000 #
    horizontal_map = {}
    #使用step1 是想遍历所有的头发丝
    step = 1 #input_strands.shape[0]/num
    if step < 1:
        step =1

    for i in range(0,input_strands.shape[0],step):
        #print i
        alpha,beta,dis = caculate_single_strand_dis(input_strands[i],vertical_range)
        if alpha<0:
            continue
        horizontal_range_index = int(alpha/horizontal_grid_range)

        if horizontal_range_index == horizontal_grid:
            horizontal_range_index = horizontal_grid-1
            if horizontal_range_index <0:
                horizontal_range_index =0
            pass
        if horizontal_range_index  in horizontal_map.keys():
            para = horizontal_map[horizontal_range_index]
            prev_alpah =para['alpha']
            prev_beta = para['beta']
            prev_dis = para['dis']
            if dis > prev_dis:
                new_para ={}
                new_para['alpha'] = alpha
                new_para['beta'] = beta
                new_para['dis'] = dis
                new_para['strand_idx'] = i
                horizontal_map[horizontal_range_index] = new_para
            else:
                continue
        else:
            new_para = {}
            new_para['alpha'] = alpha
            new_para['beta'] = beta
            new_para['dis'] = dis
            new_para['strand_idx'] = i
            horizontal_map[horizontal_range_index] = new_para
    key_indexs =[]
    for key ,para in horizontal_map.items():
        key_indexs.append(key)
    key_sort_list = list(set(key_indexs))
    return key_sort_list,horizontal_map
def get_connected_strand(input_strands,horizontal_map,connect_list):
    final_connect_list = []
    prev_start_point_pos = np.array([])
    prev_end_point_pos = np.array([])

    for i in range(0, len(connect_list)):
        final_connect_list.append([])
        c_list = connect_list[i]
        count = 0
        for j in range(0, len(c_list)):
            alpha_idx = c_list[j]
            para = horizontal_map[alpha_idx]
            strand_idx = para['strand_idx']
            strand = input_strands[strand_idx]
            v = []
            start_point_pos = strand[0][:]
            end_point_pos = strand[strand.shape[0] - 1][:]
            if count == 0:
                prev_start_point_pos = start_point_pos
                prev_end_point_pos = end_point_pos
                count += 1
                continue

            def decide_if_not_continue(prev_start_point_pos, prev_end_point_pos, start_point_pos, end_point_pos):
                start_dis = np.linalg.norm(start_point_pos - prev_start_point_pos)
                end_dis = np.linalg.norm(end_point_pos - prev_end_point_pos)
                if end_dis > 0.1:
                    print start_dis / end_dis
                    return True

            if decide_if_not_continue(prev_start_point_pos, prev_end_point_pos, start_point_pos, end_point_pos):
                final_connect_list.append([])
                final_connect_list[len(final_connect_list) - 1].append(alpha_idx)
                count = 0
            else:
                final_connect_list[len(final_connect_list) - 1].append(alpha_idx)
            prev_start_point_pos = start_point_pos
            prev_end_point_pos = end_point_pos
            count += 1
    return final_connect_list
def get_strand_vertex(strand,step):
    if len(strand) == 1:
        # print 'len(strand)',len(strand)
        return []
    v =[]
    for i in range(0, len(strand),step ):  # 最后一个点的tangent 来源于前一个
        pos = strand[i][:]
        v.append(pos)
    return  np.array(v)
def get_texture_coor(strand,i,all_num,step):
    left_u = 0
    right_u = 0.85
    cur_u = left_u+(right_u-left_u)* (all_num-1-i)/(all_num-1)
    top_v = 1.0
    bottom_v = 0

    t_v =[]
    k_range = range(0, len(strand), step)
    len_strand=len(k_range)
    for j in range(0,len(k_range)):
        cur_v = bottom_v+(top_v-bottom_v)* (len_strand-1-j)/(len_strand-1)
        t_v.append([cur_u,cur_v])
    return  np.array(t_v)
def connect_strand_to_mesh(input_strands,connect_list,horizontal_map,horizontal_grid,ourput_dir,name):
    step =4 #控制一条stand 采样点选择的步长
    all_v = np.array([])
    all_v_t = np.array([])
    all_f = np.array([])
    all_f_tf = np.array([])
    for i in range(0,len(connect_list)):
        c_list = connect_list[i]
        if len(c_list) <2:
            continue
        start_alpha_idx =c_list[0]
        start_para = horizontal_map[start_alpha_idx]
        start_strand_idx = start_para['strand_idx']
        start_strand = input_strands[start_strand_idx]
        start_strand_vtx = get_strand_vertex(start_strand,step)
        start_strand_v_t = get_texture_coor(start_strand,0,len(c_list),step)
        print start_strand_vtx.shape[0],start_strand_v_t.shape[0]
        V = start_strand_vtx
        V_T = start_strand_v_t
        prev_vtx_idx  = np.array(range(0,V.shape[0]))
        F = np.array([])

        for j in range(1,len(c_list),step):
            alpha_idx =c_list[j]
            para = horizontal_map[alpha_idx]
            strand_idx = para['strand_idx']
            strand = input_strands[strand_idx]
            strand_vtx = get_strand_vertex(strand,step)
            strand_v_t = get_texture_coor(strand, j, len(c_list),step)
            cur_vtx_idx = np.array(range(0,strand_vtx.shape[0]))
            cur_vtx_idx[:]+=V.shape[0]
            V = np.vstack((V,strand_vtx))
            V_T= np.vstack((V_T,strand_v_t))
            cur_f = []
            for k in range(0,cur_vtx_idx.shape[0]-1):
                cur_f.append([cur_vtx_idx[k],cur_vtx_idx[k+1],prev_vtx_idx[k]])
                cur_f.append([cur_vtx_idx[k + 1],prev_vtx_idx[k+1], prev_vtx_idx[k]])
            cur_f = np.array(cur_f)
            if F.size == 0:
                F = cur_f
            else:
                F = np.vstack((F,cur_f))
            prev_vtx_idx = cur_vtx_idx
        all_v,all_f = add_vertex_faces(all_v,all_f,V,F)
        all_v_t,all_f_tf = add_vertex_faces(all_v_t,all_f_tf,V_T,F)
    mesh_v = all_v
    mesh_f = all_f
    write_full_obj(mesh_v, mesh_f, np.array([]), np.array([]),all_v_t, all_f_tf, np.array([]),
                   ourput_dir + name+ '_'+str(horizontal_grid).zfill(5)+'.obj', generate_mtl=True,
                   verbose=False, img_name='hair2.png')
def write_heliex(input_strands,key_sort_list,horizontal_map,horizontal_grid,ourput_dir,name):
    mesh_v =[]
    mesh_f =[]
    mesh_v = np.array(mesh_v)
    mesh_f = np.array(mesh_f)
    mesh_n =[]
    mesh_n_f =[]
    mesh_vt = []
    mesh_vt_f =[]
    mesh_n = np.array(mesh_n)
    mesh_n_f = np.array(mesh_n_f)
    mesh_vt = np.array(mesh_vt)
    mesh_vt_f = np.array(mesh_vt_f)
    mesh_v_color = np.array([])
    for i in key_sort_list:
        para = horizontal_map[i]
        strand_idx = para['strand_idx']
        phis, thetas, points = convert_to_helix(input_strands[strand_idx])
        if len(phis) == 0:
            continue
        # print phis
        # print thetas
        v, f, vt, vt_f, v_color = convert_2_mesh(phis, thetas, points, radius=0.001, step=1, if_fix_dir=False)
        # mesh_v.append(v)
        # mesh_f.append(f)
        mesh_v, mesh_f = add_vertex_faces(mesh_v, mesh_f, v, f)
        mesh_vt, mesh_vt_f = add_vertex_faces(mesh_vt, mesh_vt_f, vt, vt_f)
        if mesh_v_color.size == 0:
            mesh_v_color = v_color
        else:
            mesh_v_color = np.vstack((mesh_v_color, v_color))
    write_full_obj(mesh_v, mesh_f, mesh_n, mesh_n_f, mesh_vt, mesh_vt_f, mesh_v_color,
                   ourput_dir + name+ '_'+str(horizontal_grid).zfill(5)+'.obj', generate_mtl=False,
                   verbose=False, img_name='hair2.png')
def get_outer_strands_wrap(input_strands,ourput_dir,name):
    horizontal_grid = 100
    for i in range(0, input_strands.shape[0]):
        strand = np.array(input_strands[i])
        strand[:,1]-=1.7178 # 归一化数值
        input_strands[i] = strand
    key_sort_list,horizontal_map = strands_key_idx(input_strands,horizontal_grid=horizontal_grid,vertical_range = pi/90)

    print '连续序列对：',continuous_str(key_sort_list)
    print 'key_set len :',len(key_sort_list)
    connect_list = split_connected_list(key_sort_list)
    final_connect_list = get_connected_strand(input_strands, horizontal_map, connect_list)
    print final_connect_list
    print len(final_connect_list)
    connect_list = final_connect_list


    #write_heliex()
    connect_strand_to_mesh(connect_list)

def get_outer_strands():
    file_dir = 'E:/workspace/dataset/hairstyles/'
    prj_dir = 'E:/workspace/dataset/hairstyles/hair/'
    ourput_dir = 'E:\workspace\dataset\hairstyles\hair/test_outer_strand/test_connect_with_texture2/'
    b = FileFilt()
    b.FindFile(dirr=prj_dir)
    count =0
    target = 1
    dd =1
    for k in b.fileList:
        if k == '':
            continue
        if count == target:
            pass
        else:
            count += 1
            #continue
        print k.split("/")[-1]
        filename_split = k.split("/")[-1].split(".")
        print filename_split
        if len(filename_split) > 1:
            #       print str(filename_split[-2])
            file_name = str(filename_split[-2])
        else:
            file_name = str(filename_split[0])
        if file_name =='strands00390':
            dd =1
            continue
            pass
        else:
            pass
            #continue
        if dd ==0:
            continue
        input_strands = read_bin(prj_dir +  file_name + '.data')
        get_outer_strands_wrap(input_strands, ourput_dir,file_name)
def get_hair_wrapper_single(input_strands,ourput_dir,name):
    from skimage import measure
    from triangle_raster import BBox3f
    from sklearn.neighbors import NearestNeighbors
    hair_point = np.array([])

    for i in range(0, input_strands.shape[0]):
        strand = np.array(input_strands[i])
        if strand.shape[0] <3:
            #print 'strand sample point too little'
            continue
        strand[:,1]-=1.7178 # 归一化数值
        input_strands[i] = strand
        if hair_point.size == 0:
            hair_point = strand
        else:
            hair_point = np.vstack((hair_point,strand))
    bb3 = BBox3f()
    bb3.addvertex_array(hair_point)
    point_rad = 2* np.linalg.norm(hair_point[0,:]-hair_point[1,:]) #使用点间距离的两倍来代替
    print bb3.min,bb3.max
    bb_min = bb3.min*1.5
    bb_max = bb3.max*1.5
    x_range = bb_max[0] - bb_min[0]
    y_range = bb_max[1] - bb_min[1]
    z_range = bb_max[2] - bb_min[2]
    min_range = min([x_range,y_range,z_range])
    x_dim = 100
    y_dim = 100
    z_dim = 100
    x_step = (bb_max[0] - bb_min[0])/x_dim
    y_step = (bb_max[1] - bb_min[1])/y_dim
    z_step = (bb_max[2] - bb_min[2])/z_dim
    min_step =  min([x_step ,y_step,z_step])
    x_dim = int((bb_max[0] - bb_min[0])/min_step)
    y_dim = int((bb_max[1] - bb_min[1])/min_step)
    z_dim = int((bb_max[2] - bb_min[2])/min_step)
#    point_rad = min_step
    grid_label = np.zeros((x_dim,y_dim,z_dim),np.float)
    grid_label[:,:,:] = 1
    # 1 代表外 -1 代表内
    neigbor = NearestNeighbors(n_neighbors=1,radius=point_rad )
    neigbor.fit(hair_point)
    from time import time




    t0 = time()
    if 0:
        for i in range(0,x_dim):
            for j in range(0,y_dim):
                for k in range(0,z_dim):
                    #偏移0.5
                    v = np.array([bb_min[0]+x_step*i+x_step/2,
                                  bb_min[1] + y_step * j + y_step / 2,
                                  bb_min[2] + z_step * k + z_step / 2
                                  ])
                    v =v.reshape(1, -1)
                    #判断v是否在最近某个采样点的影响范围内
                    distances, indices = neigbor.kneighbors(v,return_distance=True)
                    #target = hair_point[indices[0][0]]
                    #if np.linalg.norm(v - target)<point_rad:
                    if distances < point_rad:
                        grid_label[i,j,k] = -1
                        print distances,point_rad
                        pass
                    else:
                        pass
    t1 = time()
    print 'marching cube',t1-t0
    t0 = time()
    grid_pos = np.zeros((x_dim*y_dim*z_dim,3))
    for i in range(0, x_dim):
        for j in range(0, y_dim):
            for k in range(0, z_dim):
                # 偏移0.5
                v = np.array([bb_min[0] + min_step * i + x_step / 2,
                              bb_min[1] + min_step * j + y_step / 2,
                              bb_min[2] + min_step * k + z_step / 2
                              ])
                grid_pos[i*y_dim*z_dim+j*z_dim+k]=v
    t1 = time()
    print 'grid_pos ',t1-t0
    t0 = time()
    distances, indices = neigbor.kneighbors(grid_pos, return_distance=True)
    t1 = time()
#    print 'find neigbor ',t1-t0
    t0 = time()
    radius_distances, radius_indices = neigbor.radius_neighbors(grid_pos,radius = point_rad, return_distance=True)
    print  'radius_distances.shape',radius_distances.shape
    print  'radius_indices.shape',radius_indices.shape
    t1 = time()
    print 'find radius neigbor ',t1-t0
    t0 = time()
    #use_stratege  0 :not use onepoint level set 1:use one point level set 2:use multi levelset
    use_stratage =0
    def k_kernel(s):
        return  max([0,(1-s**2)**3])
    point_rad = min_range/30.0
    for i in range(0, x_dim):
        for j in range(0, y_dim):
            for k in range(0, z_dim):
                index = i * y_dim * z_dim + j * z_dim + k
                if use_stratage == 0: # use level set
                    grid_label[i, j, k] = distances[index, :] - point_rad
                elif use_stratage == 1:
                    if distances[index,:]< point_rad:
                        grid_label[i, j, k] = -1
                elif use_stratage == 2:
                    radius_num = radius_indices[index].shape[0]
                    if radius_num == 0:
                        continue
                    if radius_num>20:
                        radius_num =20
                    weight = np.zeros(radius_num)
                    weight_sum = 0.0
                    for m in range(0,radius_num):
                        weight_sum += k_kernel(radius_distances[index][m]/point_rad)
                        pass
                    for m in range(0,radius_num):
                        weight[m] = k_kernel(radius_distances[index][m]/point_rad)/weight_sum
                    x_average = 0
                    for m in range(0,radius_num):
                        x_average+=weight[m]* hair_point[radius_indices[index][m],:]
                    radius_average = 0
                    # for m in range(0,radius_num):
                    #     radius_average+=weight[m]* radius_distances[index][m]
                    # radius_average*=2
                    radius_average = max([min_step*10,10*point_rad])
                    grid_v = grid_pos[index]
                    grid_label[i, j, k] = np.linalg.norm(grid_v - x_average)-radius_average
                else:
                    pass

    t1 = time()
    print 'assign grid ',t1-t0
    verts, faces, normals, values = measure.marching_cubes_lewiner(grid_label, 0)
    for i in range(0,verts.shape[0]):
        x,y,z = verts[i, :]
        x = bb_min[0]+x/x_dim*(bb_max[0] - bb_min[0])
        y = bb_min[1]+y/y_dim*(bb_max[1] - bb_min[1])
        z = bb_min[2]+z/z_dim*(bb_max[2] - bb_min[2])
        verts[i,:] = [x,y,z]
    write_full_obj(mesh_v=verts, mesh_f=faces, mesh_n=np.array([]),mesh_n_f=np.array([]),mesh_tex=np.array([]),mesh_tex_f=np.array([]),
                   vertexcolor=np.array([]),filepath=ourput_dir+name+'range_1_10'+str(point_rad)+'.obj', generate_mtl=False,verbose=False,img_name = 'default.png')




def get_hair_wrapper():
    file_dir = 'E:/workspace/dataset/hairstyles/'
    prj_dir = 'E:/workspace/dataset/hairstyles/hair/'
    ourput_dir = 'E:\workspace\dataset\hairstyles\hair/test_hair_wrapper/'
    b = FileFilt()
    b.FindFile(dirr=prj_dir)
    count =0
    target = 1
    dd =1
    for k in b.fileList:
        if k == '':
            continue
        if count == target:
            pass
        else:
            count += 1
            #continue
        print k.split("/")[-1]
        filename_split = k.split("/")[-1].split(".")
        print filename_split
        if len(filename_split) > 1:
            #       print str(filename_split[-2])
            file_name = str(filename_split[-2])
        else:
            file_name = str(filename_split[0])
        if file_name =='strands00390':
            dd =1
            continue
            pass
        else:
            pass
            #continue
        if dd ==0:
            continue
        input_strands = read_bin(prj_dir +  file_name + '.data')
        get_hair_wrapper_single(input_strands, ourput_dir,file_name)
def smooth_hair():
    from fitting.util import laplacian_smooth
    prj_dir = 'E:\workspace\dataset\hairstyles\hair/test_hair_wrapper/hair_wrap/'
    ourput_dir = 'E:\workspace\dataset/hairstyles/hair/test_hair_wrapper/hair_smooth/'
    b = FileFilt()
    b.FindFile(dirr=prj_dir)
    count = 0
    target = 1
    dd = 1
    for k in b.fileList:
        if k == '':
            continue
        if count == target:
            pass
        else:
            count += 1
            # continue
        print k.split("/")[-1]
        filename_split = k.split("/")[-1].split(".")
        print filename_split
        if len(filename_split) > 1:
            #       print str(filename_split[-2])
            file_name = str(filename_split[0])
        else:
            file_name = str(filename_split[0])
        smooth_iter =30
        laplacian_smooth(k,ourput_dir+file_name+'_smooth_'+str(smooth_iter)+'.obj',smooth_iter)
def generate_strip_from_connect_list(input_strands,connect_list,wrap_mesh,horizontal_map):
    from sklearn.neighbors import NearestNeighbors
    #这里需假设 法向的数量与点的数量一致
    wrap_v  = wrap_mesh['wrap_v']
    wrap_f  = wrap_mesh['wrap_f']
    wrap_n  = wrap_mesh['wrap_n']
    wrap_n_f= wrap_mesh['wrap_n_f']
    hair_point = np.array([])
    for i in range(0, input_strands.shape[0]):
        strand = np.array(input_strands[i])
        if strand.shape[0] <3:
            #print 'strand sample point too little'
            continue
#        strand[:,1]-=1.7178 # 归一化数值
        input_strands[i] = strand
        if hair_point.size == 0:
            hair_point = strand
        else:
            hair_point = np.vstack((hair_point,strand))
    point_rad = 2* np.linalg.norm(hair_point[0,:]-hair_point[1,:]) #使用点间距离的两倍来代替
    neigbor = NearestNeighbors(n_neighbors=1)#,radius=point_rad
    neigbor.fit(wrap_v)

    step =1 #控制一条stand 采样点选择的步长
    all_v = np.array([])
    all_v_t = np.array([])
    all_f = np.array([])
    all_f_tf = np.array([])
    for i in range(0,len(connect_list)):
        c_list = connect_list[i]
        for j in range(0,len(c_list),1):
            alpha_idx =c_list[j]
            para = horizontal_map[alpha_idx]
            strand_idx = para['strand_idx']
            strand = input_strands[strand_idx]
            strand_v = get_strand_vertex(strand,step)
            t0 = time()
            distances, indices = neigbor.kneighbors(strand_v, return_distance=True)
            indices = indices[:,0]
            constraint_dir = wrap_n[indices]
            if check_if_contained_nan(constraint_dir):
                print 'constraint_dir contain nan'
            t1 = time()
#            print 'find neigbor ', t1 - t0
            phis, thetas, points = convert_to_helix(strand) #转化为球坐标系的表示
            if len(phis) == 0:
                print 'phis error'
                continue
            v, f, vt, vt_f, v_color = convert_2_mesh(phis, thetas, points, constraint_dir= constraint_dir,radius=0.01, step=step, if_fix_dir=True)
            all_v,all_f = add_vertex_faces(all_v,all_f,v,f)
            all_v_t,all_f_tf = add_vertex_faces(all_v_t,all_f_tf,vt,vt_f)
    mesh_v = all_v
    mesh_f = all_f
    mesh_vt = all_v_t
    mesh_vt_f = all_f_tf
    mesh_vn = np.array([])
    mesh_vn_f = np.array([])
    mesh_v_color = np.array([])
    strip_mesh = {}
    strip_mesh['mesh_v']=mesh_v
    strip_mesh['mesh_f']=mesh_f
    strip_mesh['mesh_n']=mesh_vn
    strip_mesh['mesh_n_f']=mesh_vn_f
    strip_mesh['mesh_vt']=mesh_vt
    strip_mesh['mesh_vt_f']=mesh_vt_f
    strip_mesh['mesh_v_color']= mesh_v_color
    return strip_mesh

def check_if_contained_nan(a_in):
    isconstain = False
    if len(a_in.shape) > 1:
        for i in range(0,a_in.shape[0]):
            for j in range(0,a_in.shape[1]):
                if np.isnan(a_in[i,j]):
                    isconstain = True
                    break
    else:
        for i in range(0, a_in.shape[0]):
            if np.isnan(a_in[i]):
                isconstain = True
                break
    return isconstain
def check_if_contained_all_zeros(a_in):
    for i in range(0,a_in.shape[0]):
        if np.linalg.norm(a_in[i]) < 0.000001:
            return True
def generate_strip():

    prj_dir = 'E:/workspace/dataset/hairstyles/hair/'
    smooth_wrap_hair = 'E:\workspace\dataset\hairstyles\hair/test_hair_wrapper\hair_smooth\smooth30/'
    ourput_dir = 'E:\workspace\dataset\hairstyles\hair/test_generate_strip/'
    b = FileFilt()
    b.FindFile(dirr=prj_dir)
    count =0
    target = 1
    dd =0
    for k in b.fileList:
        if k == '':
            continue
        if count == target:
            pass
        else:
            count += 1
            #continue
        print k.split("/")[-1]
        filename_split = k.split("/")[-1].split(".")
        print filename_split
        if len(filename_split) > 1:
            #       print str(filename_split[-2])
            file_name = str(filename_split[-2])
        else:
            file_name = str(filename_split[0])
        if file_name != 'strands00197' and dd:
            continue
        else:
            dd = 0
        input_strands = read_bin(prj_dir + file_name + '.data')
        horizontal_grid = 100
        for i in range(0, input_strands.shape[0]):
            strand = np.array(input_strands[i])
            strand[:, 1] -= 1.7178  # 归一化数值
            input_strands[i] = strand
        key_sort_list, horizontal_map = strands_key_idx(input_strands, horizontal_grid=horizontal_grid,
                                                        vertical_range=pi / 90)
        print '连续序列对：', continuous_str(key_sort_list)
        print 'key_set len :', len(key_sort_list)
        connect_list = split_connected_list(key_sort_list)

        smooth_hair_filepath = smooth_wrap_hair+file_name+'range_1_100_smooth_30'+'.obj'
        wrap_v, wrap_f, wrap_t, wrap_t_f, wrap_n, wrap_n_f = read_igl_obj(smooth_hair_filepath)

        if check_if_contained_all_zeros(wrap_n):
            wrap_n = get_vertex_normal(wrap_v, wrap_f)
            if check_if_contained_all_zeros(wrap_n):
                print 'wrap_n contain _all_zeros'
        if check_if_contained_nan(wrap_n):
            print 'wrap_n contain nan'
        wrap_mesh = {'wrap_v':wrap_v,'wrap_f':wrap_f,'wrap_n':wrap_n,'wrap_n_f':wrap_n_f}
        strip_mesh = generate_strip_from_connect_list(input_strands=input_strands, connect_list=connect_list,wrap_mesh=wrap_mesh,horizontal_map=horizontal_map)
        write_full_obj(strip_mesh['mesh_v'], strip_mesh['mesh_f'], strip_mesh['mesh_n'], strip_mesh['mesh_n_f'], strip_mesh['mesh_vt'],
                       strip_mesh['mesh_vt_f'], strip_mesh['mesh_v_color'],
                       ourput_dir + file_name+ '_'+str(horizontal_grid).zfill(5)+'.obj', generate_mtl=False,
                       verbose=False, img_name='hair2.png')
    pass
if __name__ == '__main__':
    #seg_dir_similarity()
    #generta_segment_map_batch('/render_hair_seg_body_high/',use_vertex_color = False)
    #generta_segment_map_batch('/render_hair_dir_body_high/', use_vertex_color=True)
#    generate_polystrip_mesh_with_dir_color()
    #caculate_hair_seg_bin_batch()

    #print hairs_seg_bin
    #generate_hair_dir_single()
    # caculate_hair_seg_bin_batch("E:/workspace/dataset/hairstyles/hair/convert_hair_dir/render_hair_seg_body/",
    #                             "E:/workspace/dataset/hairstyles/hair/convert_hair_dir/render_hair_dir_body/",
    #                             "E:/workspace/dataset/hairstyles/hair/convert_hair_dir/seg_bin/",
    #                             "E:/workspace/dataset/hairstyles/hair/convert_hair_dir/dir_bin/"
    #                             )
    #test_scale_bbox()
    #build_hair_for_img_batch()
#get_outer_strands()
    #get_hair_wrapper()
#    smooth_hair()
    generate_strip()


# write_simple_obj(v,f,file_dir+'/obj/'+'strands00002'+'_'+str(stand_num)+'.obj')
# v =[]
# f=[]
# for i in range(0,len(result)):
#     for j in range(0,len(result[i])):
#         vv = result[i][j]
#         v.append(vv)
#write_simple_obj(v,f,file_dir+'/obj/'+'strands00002'+'.obj')


#print result

