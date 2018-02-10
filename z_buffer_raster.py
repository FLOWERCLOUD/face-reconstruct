# -- coding: utf-8 --
from triangle_raster import  BBoxf,BBoxi,MetroMesh,RasterSampler,ClosestPoint,BBox3f
import numpy as np
from numpy.linalg import norm as norm
import cv2

import  math
def SingleFaceRasterZbuffer(mesh,f,ps,v0,v1,v2,correctSafePointsBaryCoords):
    FLT_MAX = 3.40282346638528860e+38

    bboxf = BBoxf(v0,v1,v2)
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
    b0  = (bbox.min[0]-v0[0])*d10[1] - (bbox.min[1]-v0[1])*d10[0]
    b1  = (bbox.min[0]-v1[0])*d21[1] - (bbox.min[1]-v1[1])*d21[0]
    b2  = (bbox.min[0]-v2[0])*d02[1] - (bbox.min[1]-v2[1])*d02[0]
    # Preparazione degli steps
    db0 = d10[1]
    db1 = d21[1]
    db2 = d02[1]
    # Preparazione segni
    dn0 = -d10[0]
    dn1 = -d21[0]
    dn2 = -d02[0]
    #Calculating orientation
    flipped = bool(not (d02[0:2].dot(np.array([-d10[1], d10[0]])) >= 0))
    # Calculating border edges
    borderEdges = np.zeros((3,6))
    edgeLength = np.array([0.0,0.0,0.0])
    edgeMask = 0
    def edge_length(edge):
        v0 = edge[0:2]
        v1 = edge[2:4]
        return norm(v0-v1)
    if (mesh.IsBorder_V(f,0)) :
        borderEdges[0,:] = np.hstack([v0, v1])
        edgeLength[0] = edge_length(borderEdges[0,:])
        edgeMask |= 1

    if (mesh.IsBorder_V(f,1)) :
        borderEdges[1,:] = np.hstack([v1, v2])
        edgeLength[1] = edge_length(borderEdges[1,:])
        edgeMask |= 2

    if (mesh.IsBorder_V(f,2)) :
        borderEdges[2,:] =np.hstack([v2, v0])
        edgeLength[2] = edge_length(borderEdges[2,:])
        edgeMask |= 4
    # Rasterizzazione
    de = v0[0]*v1[1]-v0[0]*v2[1]-v1[0]*v0[1]+v1[0]*v2[1]-v2[0]*v1[1]+v2[0]*v0[1]

    for  x  in range(bbox.min[0]-1,bbox.max[0]+1+1):
        bool_in = False
        n  = [b0-db0-dn0, b1-db1-dn1, b2-db2-dn2]
        for y  in range(bbox.min[1]-1,bbox.max[1]+1+1):

            if( ((n[0]>=0 and n[1]>=0 and n[2]>=0) or (n[0]<=0 and n[1]<=0 and n[2]<=0))  and (de != 0)):

                baryCoord = np.array([0.0,0.0,0.0])
                baryCoord[0] =  np.float(-y*v1[0]+v2[0]*y+v1[1]*x-v2[0]*v1[1]+v1[0]*v2[1]-x*v2[1])/de
                baryCoord[1] = -np.float( x*v0[1]-x*v2[1]-v0[0]*y+v0[0]*v2[1]-v2[0]*v0[1]+v2[0]*y)/de
                baryCoord[2] = 1-baryCoord[0]-baryCoord[1]
                z = np.dot(baryCoord,np.array([v0[2],v1[2],v2[2]]))
                ps.AddRenderSample(mesh,f, baryCoord, [x,y,z], 0)
                bool_in = True
            else:
                # Check whether a pixel outside (on a border edge side) triangle affects color inside it
                px = np.array([float(x),float(y)])
                closePoint = np.array([0.0,0.0])
                closeEdge = -1
                minDst = FLT_MAX

                # find the closest point (on some edge) that lies on the 2x2 squared neighborhood of the considered point
                for i in range(0,3):
                    if (edgeMask & (1 << i)):
                        if ( ((not flipped) and (n[i]<0)) or
                             (  flipped  and (n[i]>0))   ):
                            close = ClosestPoint(borderEdges[i, :], px)
                            dst = norm( close-px)
                            if(dst < minDst and
                               close[0] > px[0]-1 and close[0] < px[0]+1 and
                               close[1] > px[1]-1 and close[1] < px[1]+1) :
                                minDst = dst
                                closePoint = close
                                closeEdge = i

                if (closeEdge >= 0):
                    baryCoord = np.array([0.0, 0.0, 0.0])
                    if (correctSafePointsBaryCoords):
                        #Add x,y sample with closePoint barycentric coords (on edge)
                        baryCoord[closeEdge] = norm(closePoint - borderEdges[closeEdge,2:4])/edgeLength[closeEdge]
                        baryCoord[(closeEdge+1)%3] = 1 - baryCoord[closeEdge]
                        baryCoord[(closeEdge+2)%3] = 0
                    else:
                        # Add x,y sample with his own barycentric coords (off edge)
                        baryCoord[0] =  np.float(-y*v1[0]+v2[0]*y+v1[1]*x-v2[0]*v1[1]+v1[0]*v2[1]-x*v2[1])/de
                        baryCoord[1] = - np.float( x*v0[1]-x*v2[1]-v0[0]*y+v0[0]*v2[1]-v2[0]*v0[1]+v2[0]*y)/de
                        baryCoord[2] = 1-baryCoord[0]-baryCoord[1]
                    z = np.dot(baryCoord, np.array([v0[2], v1[2], v2[2]]))
                    ps.AddRenderSample(mesh,f, baryCoord, [x,y,z], minDst)
                    bool_in = True

            n[0] += dn0
            n[1] += dn1
            n[2] += dn2

        b0 += db0
        b1 += db1
        b2 += db2

class RasterSampler_Zbuffer:

    def __init__(self, image,zbuffer,harmonic_coeff=[]):
        self.image = image
        self.zbuffer= zbuffer
        self.harmonic_coeff = harmonic_coeff
        pass

    # expects points outside face (affecting face color) with edge distance > 0
    # f Nx3 ,p 重心坐标， tp 2d坐标，左下角为原点
    def AddRenderSample(self, mesh, f_id, p, tp,edgeDist=0.0):
        c = []  # 重心坐标对应的颜色值

        height, width, dim = self.image.shape
        #    print p
        p = p.reshape(p.size, 1)
    #    tp = [math.floor(v[0]),math.floor(v[1])]
        z = tp[2]
        if height-tp[1]>=height or tp[0] >=width:
            #print  height-tp[1],tp[0]
            return
        if height-tp[1]<0 or tp[0] <0:
            #print  height-tp[1],tp[0]
            return

        if z > self.zbuffer[height-tp[1], tp[0], 0]: #通过zbuffer测试
            #简单地线性插值,暂时不计算光照
            v0_color = mesh.v_color(f_id, 0)
            v1_color = mesh.v_color(f_id, 1)
            v2_color = mesh.v_color(f_id, 2)
            v0_color = v0_color.reshape(v0_color.size, 1)
            v1_color = v1_color.reshape(v1_color.size, 1)
            v2_color = v2_color.reshape(v2_color.size, 1)
            m = np.hstack([v0_color, v1_color, v2_color])
            c =np.dot(m, p)
            self.zbuffer[height-tp[1], tp[0], :] = z
            self.image[height-tp[1], tp[0], :] = [c[2], c[1], c[0], 255]

#MetroMesh & m VertexSampler &ps
def Render(m, ps, imageWidth,imageHeight,correctSafePointsBaryCoords=True):

        print "Similar Triangles face sampling\n"
        bbox = BBox3f()
        bbox.addvertex_array(m.v)

        x_range = bbox.max[0] - bbox.min[0]
        y_range = bbox.max[1] - bbox.min[1]
        z_range = bbox.max[2] - bbox.min[2]
        bbox.max[0]+=x_range*0.1
        bbox.min[0]-=x_range*0.1
        bbox.max[1]+=y_range*0.1
        bbox.min[1]-=y_range*0.1
        bbox.max[2]+=z_range*0.1
        bbox.min[2]-=z_range*0.1
        x_range = bbox.max[0] - bbox.min[0]
        y_range = bbox.max[1] - bbox.min[1]
        z_range = bbox.max[2] - bbox.min[2]

        x_step = (imageWidth)/x_range
        y_step = (imageHeight) /y_range
        #取最小步长
        min_step = min([x_step,y_step])
        x_step = min_step
        y_step = min_step
        z_step = 256.0 / z_range

        for fi in range(0,m.face.shape[0]):
            ti = np.zeros([3,3])
            for i in range(0,3):
                x = (m.f_v(fi, i)[0] - bbox.min[0])*x_step
                y = (m.f_v(fi, i)[1] - bbox.min[1])*y_step
                z = (m.f_v(fi, i)[2] - bbox.min[2])*z_step
                ti[i] = np.array([x,y,z])
            SingleFaceRasterZbuffer(m,fi, ps, ti[0,:], ti[1,:], ti[2,:], correctSafePointsBaryCoords)


def Render_withmybbox(m, ps, imageWidth, imageHeight, bbox,correctSafePointsBaryCoords=True):
    print "Similar Triangles face sampling\n"
    x_range = bbox.max[0] - bbox.min[0]
    y_range = bbox.max[1] - bbox.min[1]
    z_range = bbox.max[2] - bbox.min[2]

    x_step = float(imageWidth) / x_range
    y_step = float(imageHeight) / y_range
    # 取最小步长
    #min_step = min([x_step, y_step])
    #x_step = min_step
    #y_step = min_step
    z_step = 256.0 / z_range

    for fi in range(0, m.face.shape[0]):
        ti = np.zeros([3, 3])
        for i in range(0, 3):
            x = (m.f_v(fi, i)[0] - bbox.min[0]) * x_step
            y = (m.f_v(fi, i)[1] - bbox.min[1]) * y_step
            z = (m.f_v(fi, i)[2] - bbox.min[2]) * z_step
            ti[i] = np.array([x, y, z])
        SingleFaceRasterZbuffer(m, fi, ps, ti[0, :], ti[1, :], ti[2, :], correctSafePointsBaryCoords)

def Mesh_render_to_image(file_path,mesh,textW,textH,overwrite = True,assign = True,pp =True):
    image = np.zeros((textH,textW,4),np.uint8) #BGRA
    image[:,:,:] =[0,0,0,0]
    zbuffer = np.zeros((textH, textW, 1), np.float32)  # BGRA

    # 建立 vt 的连接关系
    #0 代表 v0-v1 , 1 代表 v1-v2 ,2 代表 v1-v2
#    border_flags = np.zeros( mesh.num_texture_face,3)

    #  Compute (texture-space) border edges
    from time import time
    def print_para_resut(step,timer_end,timer_start):
        print step
        print "in %f sec\n" % (timer_end - timer_start)
    timer_start = time()
    mesh.FaceFace()
    timer_end = time()
    print_para_resut('FaceFaceFromTexCoord()', timer_end, timer_start)

    timer_start = time()
    ps = RasterSampler_Zbuffer(image,zbuffer)
    Render(mesh, ps, textW,textH,True)
    timer_end = time()
    print_para_resut('Texture()', timer_end, timer_start)
    count = 0
    for hieght in range(0,textH):
        for width in range(0,textW):
            pixel = image[hieght, width, :]
            if pixel[3] < 255 and pixel[3] >0:
#                print pixel[3]
                pixel[3] =255
                count+=1
            if pixel[3] < 255 :
                pixel[3] =255

    cv2.imwrite(file_path, image)

def Mesh_render_to_image_withmy_bbox(file_path,mesh,textW,textH,bbox):
    image = np.zeros((textH,textW,4),np.uint8) #BGRA
    image[:,:,:] =[0,0,0,0]
    zbuffer = np.zeros((textH, textW, 1), np.float32)  # BGRA

    # 建立 vt 的连接关系
    #0 代表 v0-v1 , 1 代表 v1-v2 ,2 代表 v1-v2
#    border_flags = np.zeros( mesh.num_texture_face,3)

    #  Compute (texture-space) border edges
    from time import time
    def print_para_resut(step,timer_end,timer_start):
        print step
        print "in %f sec\n" % (timer_end - timer_start)
    timer_start = time()
    mesh.FaceFace()
    timer_end = time()
    print_para_resut('FaceFaceFromTexCoord()', timer_end, timer_start)

    timer_start = time()
    ps = RasterSampler_Zbuffer(image,zbuffer)
    Render_withmybbox(mesh, ps, textW,textH,bbox,True)
    timer_end = time()
    print_para_resut('Texture()', timer_end, timer_start)
    count = 0
    for hieght in range(0,textH):
        for width in range(0,textW):
            pixel = image[hieght, width, :]
            if pixel[3] < 255 and pixel[3] >0:
#                print pixel[3]
                pixel[3] =255
                count+=1
            if pixel[3] < 255 :
                pixel[3] =255

    cv2.imwrite(file_path, image)
