import numpy as np
import plotly.offline as py
import plotly.graph_objs as go
import itertools
import common
import networkx as nx

MARKERS_MINE_2_PLOTLY = {'x':'x', '.':'dot', 'o':'circle-open'}
LINETYPE_MINE_2_PLOTLY = {'--':'dash', '-.':'dashdot','.':'dot'}
# responsible for plotting things on screen
class Plotter(object):
    def __init__(self):
        self.graph_objs = []
        self.aabb = None #bounding box of data, aabb[0]=min corner, aabb[1]=max corner
        
    def update_aabb(self, pts):
        minc = pts.min(axis=0)
        maxc = pts.max(axis=0)
        if self.aabb is None:
            self.aabb = np.array([minc,maxc])
        else:
            minc = np.minimum(self.aabb[0], minc)
            maxc = np.maximum(self.aabb[1], maxc)
            self.aabb = np.array([minc,maxc])
    
    def addPoints(self, pts, color = None, marker = None, marker_size = None, name = None):
        ''' add scatter of points
        '''
        pts = vec1n_to_mat1n(pts)
            
        marker_style = dict()
        if color is not None:
            marker_style['color'] = color2string(color)
        if marker is not None:
            marker_style['symbol'] = MARKERS_MINE_2_PLOTLY[marker]
        if marker_size is not None:
            marker_style['size'] = marker_size
            
        if pts.shape[1]==3:
            obj = go.Scatter3d(
                x=pts[:,0], y=pts[:,1],z=pts[:,2],
                mode = 'markers', showlegend= False)
        else:
            obj = go.Scatter(
                x=pts[:,0], y=pts[:,1], mode = 'markers')
        
        if len(marker_style) >0:
            obj['marker'] = marker_style
            
        if name is not None:
            obj['name'] = name
            obj['showlegend'] = True
        
        self.graph_objs.append(obj)
        self.update_aabb(pts)
    
    def addPolyline(self, pts, color = None, line_width = None, line_type = None,
                    marker = None, name = None, marker_size = None, alpha = 1.0):
        ''' add polyline object to the plot
        
        pts = the points defining the polyline
        
        color = (r,g,b,a) color where each component is within [0,1]. For example, (1,0,0,0) is red. Or (r,g,b) similarly.
        '''        
        pts = vec1n_to_mat1n(pts)
        line_style = dict()
        marker_style = dict()
        
        if color is not None:
            line_style['color'] = (color2string(color))
        if line_width is not None:
            line_style['width'] = line_width
        if line_type is not None and line_type!='-':
            line_style['dash'] = LINETYPE_MINE_2_PLOTLY[line_type]
            
        if marker is not None:
            marker_style['symbol'] = MARKERS_MINE_2_PLOTLY[marker]
        if marker_size is not None:
            marker_style['size'] = marker_size
            
        mode_str = 'lines'
        if len(marker_style) > 0:
            mode_str += '+markers'
            
        if pts.shape[1] == 3:
            obj = go.Scatter3d(
                x=pts[:,0], y=pts[:,1],z=pts[:,2],
                mode=mode_str, showlegend = False
            )
        else:
            obj = go.Scatter(
                x=pts[:,0], y=pts[:,1],
                mode=mode_str, showlegend = False
            )            
        
        if alpha is not None:
            obj['opacity'] = alpha
        
        if len(line_style) >0:
            obj['line'] = line_style
        if len(marker_style) >0:
            obj['marker'] = marker_style
        
        if name is not None:
            obj['name'] = name
            obj['showlegend'] = True
            
        self.graph_objs.append(obj)
        self.update_aabb(pts)
        
    def addTriMesh(self, vertices, faces, alpha = None, 
                   show_face = True, face_color = None,  face_alpha = None,
                   show_edge = False, edge_color = None, edge_alpha = None, 
                   line_width = None, name = None):
        
        face_alpha = alpha if alpha is not None else face_alpha
        edge_alpha = alpha if alpha is not None else edge_alpha
        self.update_aabb(vertices)
        
        if show_face:            
            obj_face = go.Mesh3d(
                x=vertices[:,0], y=vertices[:,1],z=vertices[:,2],
                i=faces[:,0], j=faces[:,1], k=faces[:,2])
            
            if face_color is not None:
                obj_face['color'] = color2string(face_color)
            if face_alpha is not None:
                obj_face['opacity'] = face_alpha
                
            if alpha is not None:
                obj_face['opacity'] = alpha                
                
            if name is not None:
                obj_face['name'] = name
                obj_face['showlegend'] = True
            else:
                obj_face['showlegend'] = False
            self.graph_objs.append(obj_face)
        
        import common.shortfunc as sf
        if show_edge:
            e1 = faces[:,[0,1]]
            e2 = faces[:,[0,2]]
            e3 = faces[:,[1,2]]
            es = np.row_stack([e1,e2,e3])
            es = np.sort(es, axis=1)
            es_unique = sf.unique_rows(es)[0]
            p1 = vertices[es_unique[:,0]]
            p2 = vertices[es_unique[:,1]]
            self.addLineSegments(p1,p2,color=edge_color, 
                                 line_width=line_width, 
                                 alpha = edge_alpha,
                                 name = name)
            
        #if show_edge:
            #G = nx.Graph()
            #G.add_nodes_from(np.arange(len(vertices)))
            #G.add_edges_from(faces[:,[0,1]])
            #G.add_edges_from(faces[:,[0,2]])
            #G.add_edges_from(faces[:,[1,2]])
            #G.remove_nodes_from(nx.isolates(G))
            #for nodes in nx.connected_components(G):
                #sg = G.subgraph(nodes)
                #sg = nx.DiGraph(sg)
                #epath = np.array(list(nx.eulerian_circuit(sg)))
                #idx = epath[:,0]
                #idx = np.append(idx,0)
                #pts = vertices[idx]
                #self.addPolyline(pts, color = edge_color, line_width = line_width, 
                                 #alpha = edge_alpha, name=name)
            
        
    def addLineSegments(self,pts1,pts2, color = None, line_width = None, line_type = None,
                    marker = None, name = None, marker_size = None, alpha = 1.0):
        pts1 = vec1n_to_mat1n(pts1)
        pts2 = vec1n_to_mat1n(pts2)
        self.update_aabb(pts1)
        self.update_aabb(pts2)
        
        ndim = pts1.shape[1]
        xs = []
        ys = []
        zs = []            
        for p, q in zip(pts1,pts2):
            xs += [p[0],q[0],None]
            ys += [p[1],q[1],None]
            if len(p)>2:
                zs += [p[2],q[2],None]
                
        line_style = dict()
        marker_style = dict()
        
        if color is not None:
            line_style['color'] = (color2string(color))
        if line_width is not None:
            line_style['width'] = line_width
        if line_type is not None and line_type!='-':
            line_style['dash'] = LINETYPE_MINE_2_PLOTLY[line_type]
            
        if marker is not None:
            marker_style['symbol'] = MARKERS_MINE_2_PLOTLY[marker]
        if marker_size is not None:
            marker_style['size'] = marker_size
            
        mode_str = 'lines'
        if len(marker_style) > 0:
            mode_str += '+markers'        
            
        if ndim == 2:
            obj = go.Scatter(x=xs,y=ys,mode=mode_str)
        else:
            obj = go.Scatter3d(x=xs,y=ys,z=zs, mode=mode_str)
        if alpha is not None:
            obj['opacity'] = alpha
        
        if len(line_style) >0:
            obj['line'] = line_style
        if len(marker_style) >0:
            obj['marker'] = marker_style
        
        if name is not None:
            obj['name'] = name
            obj['showlegend'] = True        
        self.graph_objs.append(obj)
        #else:
            #for p, q in zip(pts1,pts2):
                #x = [p[0],q[0]]
                #y = [p[1],q[1]]
                #z = [p[2],q[2]]
                #obj = go.Scatter3d(x=x,y=y,z=z,mode='lines', showlegend = False)
                #self.graph_objs.append(obj)
            
        #return [obj]
        
    def is_2d(self):
        if self.aabb is None:
            return True
        else:
            return self.aabb.shape[1] == 2
    
    def show(self, filename = 'templot.html', longpix = 1000, axis_equal = False,
             show_legend = True):
        ''' longpix = number of pixels along the longer dimension, only useful when axis_equal is True
        
        axis_equal = should we use equal axis? If used, then longpix can be used to
        control the image size
        '''
        layout = go.Layout()
        boxlens = self.aabb[1]-self.aabb[0]
        imgsize = boxlens * float(longpix)/ max(boxlens)
        if axis_equal:
            xaxis = {'range': self.aabb[:,0].flatten()}
            yaxis = {'range': self.aabb[:,1].flatten()}
            layout['xaxis'] = xaxis
            layout['yaxis'] = yaxis
            
            if self.is_2d():
                layout['width'] = imgsize[0]
                layout['height'] = imgsize[1]
                
            if not self.is_2d(): #3d
                zaxis = {'range':self.aabb[:,2].flatten()}
                #layout['zaxis'] = zaxis
                asp = boxlens/boxlens.max()
                asp[asp < 1e-6] = 1e-6
                layout['scene'] = {'aspectratio':{'x':asp[0], 'y':asp[1], 'z':asp[2]}}
                
        if show_legend is not None:
            layout['showlegend'] = show_legend
        
        fig = go.Figure(data = self.graph_objs, layout = layout)
        py.plot(fig, filename = filename)
        
def color2string(color_float):
    vals = np.array(color_float)
    vals[vals>1] = 1.0
    vals[vals<0] = 0.0
    cc = np.zeros(4,dtype=float)
    cc[-1] = 1.0
    cc[:len(vals)] = vals
    s = 'rgba({0},{1},{2},{3})'.format(cc[0],cc[1],cc[2],cc[3])
    return s
        

def vec1n_to_mat1n(x):
    if len(x.shape)==1:
        return x.reshape((1,-1))
    else:
        return x