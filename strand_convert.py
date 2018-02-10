# -- coding: utf-8 --
import array
import numpy as np
from numba import jit
from fitting.util import write_simple_obj,add_vertex_faces,FileFilt,write_full_obj,readVertexColor,write_landmark_to_obj
import  math
from math import sin, cos,atan
from time import time

g_div = 2
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
    strand[:,1] -=1.7178 # 归一化数值
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
    dot_resut = np.dot(Eb, cast_stran_dir) / np.linalg.norm(Eb) / np.linalg.norm(cast_stran_dir)
    rot_theta = math.acos(dot_resut)
    if (dot_resut > 0.999):
        pass
    else:
        cross_result = np.cross(Eb, cast_stran_dir)
        if np.dot(cross_result, Et) < 0:
            # 说明角度应小于180度
            pass
        else:
            # 说明角度应大于180度
            if rot_theta < math.pi:
                rot_theta = 2 * math.pi - rot_theta
    return rot_theta


def get_color_from2d_dir(dir):
    from math import pi
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
    return color
def convert_2_mesh(phis,thetas,points , radius = 0.005,step =9):

    phis = np.array(phis)
    thetas = np.array(thetas)
    points =  np.array(points)
    v =[]
    f =[]
    vt =[]
    vt_f =[]
    v_color=[]
    global g_div
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
        stran_dir = node / np.linalg.norm(node)
        Et  = np.array([cos(phi)*cos(theata),cos(phi)*sin(theata),sin(phi)])
        En = np.array([-sin(theata),cos(theata),0])
        Eb = np.array([-sin(phi)*cos(theata),-sin(phi)*sin(theata),cos(phi)])
        rot = cacaulate_rot(stran_dir, Et, En, Eb)

        for rad in angles:
            rad -=rot # 对角度进行修正
            v_c = node+(cos(rad)* En + sin(rad)*Eb)*radius
            v.append(v_c)
        if g_div == 2:
            vt_tmp =[1.0, 1.0-ratio]
            vt.append(vt_tmp)
            vt_tmp =[0.8, 1.0-ratio]
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


        # if i == points.shape[0]-1:
        #     f.append([v0,v1,v5])
        #     f.append([v1,v2,v5])
        #     f.append([v2,v3,v5])
        #     f.append([v3,v4,v5])
    return np.array(v),np.array(f),np.array(vt),np.array(vt_f),np.array(v_color)
#@jit(nopython=True, cache=True)
def convert_batch(result,num = 100):
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
        v,f,vt,vt_f,v_color = convert_2_mesh(phis,thetas,points)
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

        v, f,n,nf,vt,vt_f,vt_color = convert_batch(result, num=stand_num)
        write_full_obj(v, f, n, nf, vt, vt_f, vt_color, prj_dir+'/convert_hair_dir/'+file_name+'_'+str(stand_num)+'.obj', generate_mtl=False,
                       verbose=False, img_name='hair2.png')
        #write_simple_obj(v,f,prj_dir+'/convert_hair/'+file_name+'_'+str(stand_num)+'.obj')
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
    sys.path.insert(0, "D:/mprojects/libiglfull/libigl/python/build/x64/Release")
    import pyigl as igl
    import numpy as np
    from triangle_raster import MetroMesh
    from fitting.util import  read_igl_obj,add_vertex_faces
    from fitting.landmarks import  load_embedding
    hair_file_dir = "G:/yuanqing/faceproject/hairstyles/hairstyles/"+ "hair/convert_hair_dir/"#"hair/convert_hair/"
    head_filr_dir = "G:/yuanqing/faceproject/hairstyles/hairstyles/"
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

    def mesh_points_by_barycentric_coordinates(mesh_verts, mesh_faces, lmk_face_idx, lmk_b_coords):
        """ function: evaluation 3d points given mesh and landmark embedding
        """
        dif1 = np.vstack([(mesh_verts[mesh_faces[lmk_face_idx], 0] * lmk_b_coords).sum(axis=1),
                          (mesh_verts[mesh_faces[lmk_face_idx], 1] * lmk_b_coords).sum(axis=1),
                          (mesh_verts[mesh_faces[lmk_face_idx], 2] * lmk_b_coords).sum(axis=1)]).T
        return dif1

    face_landmark_3d = mesh_points_by_barycentric_coordinates(head_v, head_v_f, lmk_face_idx, lmk_b_coords)
    if 0:
        model_path = './models/male_model.pkl'  # change to 'female_model.pkl' or 'generic_model.pkl', if needed
        from smpl_webuser.serialization import load_model
        model = load_model(model_path)

        #write_simple_obj(model.r,model.f,hair_file_dir + output_name + 'frame' + '.obj')
        write_landmark_to_obj(hair_file_dir + output_name + 'landmark2' + '.obj',face_landmark_3d,size = 3)
    #这里的landmark没有包括轮廓
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
        Mesh_render_to_image_withmy_bbox(hair_file_dir + output_name + file_name + '.png', mesh, image_width,
                                         image_height, bbox)
        #Mesh_render_to_image(hair_file_dir + '/render_seg/'+file_name+'.png', mesh, 256, 256)
        # write_simple_obj(v,f,prj_dir+'/convert_hair/'+file_name+'_'+str(stand_num)+'.obj')
        count += 1
def caculate_EMD_distance_1d(H1,H2):
    import sys
    sys.path.insert(0, "D:/mprojects/EmdL1_v3/EMD/x64/Release")
    import  EMD_PYTHON
    EMD_PYTHON.EMD_1D(H1, H2)
def caculate_hair_seg_bin(image,center):
    seg_image = np.zeros( (image.shape[0],image.shape[1],image.shape[2]))
    image[:,:,:] = image[::-1,:,:] #以左下角为原点
    image[:, :, :] = image[:, :, ::-1] #转化为RGB
    center_x = center[0]
    center_y = image.shape[0]-1-center[1]
    center = np.array([center_x,center_y])
    theta_array = np.linspace(0.0, 180.0, 32, False)

    for i in range(0,image.shape[0]):
        for j in range(0,image.shape[1]):
            y= i
            x = j
            p = np.array([x,y])
            dir_vector = p- center
            if np.linalg.norm(dir_vector) < 4 :
                continue
            if image[i,j,0] < 2.0 and image[i,j,1] >250.0 and image[i,j,2]<2.0:
                continue
            if image[i,j,0] < 2.0 and image[i,j,1] <2.0 and image[i,j,2]<2.0:
                continue
            seg_image[i,j] = get_color_from2d_dir(dir_vector)

    seg_image[:,:,:] = seg_image[::-1,:,:]
    seg_image[:, :, :] = seg_image[:, :, ::-1]
    return seg_image
def caculate_hair_seg_bin_batch(input_seg,input_dir,out_seg_bin,out_dir_bin):
    import  cv2

    b = FileFilt()
    b.FindFile(dirr=input_seg)
    count = 0
    target = 1

    for k in b.fileList:
        if k == '':
            continue
        print k.split("/")[-1]
        #print k.split("/")
        filename_split = k.split("/")[-1].split(".")
        print filename_split
        if len(filename_split) > 1:
            #       print str(filename_split[-2])
            file_name = str(filename_split[-2])
        else:
            file_name = str(filename_split[0])
        img = cv2.imread(k,cv2.IMREAD_COLOR )
        y_scale_up = 4.0
        y_scale_down = 20.0

        seg_img = caculate_hair_seg_bin(img,[(img.shape[1]-1)/2.0,(img.shape[0]-1)/(1+y_scale_down/y_scale_up)])
        cv2.imwrite(out_seg_bin+file_name
                    + '.png', seg_img)
        #cv2.imwrite(out_dir_bin+file_name
        #            + '.png', seg_img)



if __name__ == '__main__':
    generta_segment_map_batch('/render_hair_seg_body/',use_vertex_color = False)
    generta_segment_map_batch('/render_hair_dir_body/', use_vertex_color=True)
    #generate_polystrip_mesh_with_dir_color()
    #caculate_hair_seg_bin_batch()
    caculate_hair_seg_bin_batch("G:/yuanqing/faceproject/hairstyles/hairstyles/hair/convert_hair_dir/render_hair_seg_body/",
                                "G:/yuanqing/faceproject/hairstyles/hairstyles/hair/convert_hair_dir/render_hair_dir_body/",
                                "G:/yuanqing/faceproject/hairstyles/hairstyles/hair/convert_hair_dir/seg_bin/",
                                "G:/yuanqing/faceproject/hairstyles/hairstyles/hair/convert_hair_dir/dir_bin/"
                                )
    pass


# write_simple_obj(v,f,file_dir+'/obj/'+'strands00002'+'_'+str(stand_num)+'.obj')
# v =[]
# f=[]
# for i in range(0,len(result)):
#     for j in range(0,len(result[i])):
#         vv = result[i][j]
#         v.append(vv)
#write_simple_obj(v,f,file_dir+'/obj/'+'strands00002'+'.obj')


#print result

