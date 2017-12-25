import numpy as np
import mayavi.mlab as mlab
import mayavi.core.scene as msc
import scipy.sparse as sparse
import scipy.spatial as _spatial
import itertools

str2color = {'r':(1.,0,0), 'g':(0.,1,0), 'b':(0.,0,1), 'w':(1.,1,1), 'k':(0.,0,0)}

def augment_points(pts, augval = 0):
    return np.column_stack((pts, np.ones(len(pts)) * augval))

def makebox(xyzmin, xyzmax):
    ''' create a box model.
    
    return (vertices, faces) of a box
    '''
    xyzmin = np.array(xyzmin)
    xyzmax = np.array(xyzmax)
    cc = np.array([[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]]).astype(float)
    pts = cc * (xyzmax - xyzmin) + xyzmin
    ch = _spatial.ConvexHull(pts)
    return (pts, ch.simplices)

def vec1n_to_mat1n(x):
    x = np.atleast_2d(x)
    return x
    #if len(x.shape)==1:
        #return x.reshape((1,-1))
    #else:
        #return x

def show():
    mlab.show()

def makefig():
    return mlab.figure()

def volume(arr3d, show_box = False, box_color3f = None):
    if arr3d.dtype != float:
        arr3d = arr3d.astype(float)
    box_color3f = get_color_value(box_color3f)
    
    mlab.pipeline.volume(mlab.pipeline.scalar_field(arr3d))
    if show_box:
        boxvts, boxfaces = makebox([0,0,0], np.array(arr3d.shape)-1)
        trisurf(boxvts,boxfaces,rendertype='wireframe',color3f=box_color3f)

def scatter(pts, alpha = 1.0, mode = None, 
            scale = 1.0, linewidth=1.0, color3f = None):
    ''' mode = '2dcircle', '2dcross','2ddash','2ddiamond',
    '2dsquare','2dvertex','arrow','axes','cone','cube',
    'point','sphere'
    
    color3f = a single color, or nx3 matrix each row being a color for a point,
    the values are all within [0,1]
    '''
    pts = np.atleast_2d(pts)    
    if pts.shape[-1] == 2:
        pts = augment_points(pts)
        
    ad = {}
    ad['opacity'] = alpha
    ad['scale_factor'] = scale
    ad['line_width'] = linewidth
    if mode is not None:
        ad['mode'] = mode
        
    if type(color3f) is np.ndarray and len(color3f.shape)==2:
        assert len(color3f) == len(pts), 'color must be the same number of points'
        s = np.arange(len(pts))
        ad['scale_mode'] = 'none' #so that the scalar only affects color
        
        obj = mlab.points3d(pts[:,0], pts[:,1], pts[:,2], s, **ad)
        
        # create lut
        obj.glyph.color_mode = 'color_by_scalar'
        if color3f.shape[-1] == 3:
            ctable = np.column_stack((color3f, np.ones(len(color3f)) * alpha))
        else:
            ctable = color3f
            
        obj.module_manager.scalar_lut_manager.lut.number_of_colors = ctable.shape[0]
        obj.module_manager.scalar_lut_manager.lut.table = np.array(ctable * 255).astype('uint8')
    else:        
        color3f = get_color_value(color3f)
        if color3f is not None:
            ad['color'] = tuple(color3f)
        mlab.points3d(pts[:,0],pts[:,1],pts[:,2], **ad)
    
def plot(pts, alpha = 1.0, linewidth = 1.0, color3f = None, rendertype = 'wireframe'):
    pts = np.atleast_2d(pts)
    color3f = get_color_value(color3f)
    ad = {}
    ad['opacity'] = alpha
    ad['line_width'] = linewidth
    if rendertype == 'surface':
        ad['tube_radius'] = linewidth * 0.1
    ad['representation'] = rendertype
    if color3f is not None:
        ad['color'] = tuple(color3f)
        
    mlab.plot3d(pts[:,0],pts[:,1],pts[:,2], **ad)
    
def show_axis(origin, xyzdir, axislength=1.0, linewidth = 1.0, alpha=1.0):
    origin = np.array(origin)
    xyzdir = np.array(xyzdir)
    pts1 = np.tile(origin,(3,1))
    pts2 = pts1 + xyzdir * axislength
    colors = np.eye(3)
    for i in range(3):
        plot(np.array([pts1[i],pts2[i]]), alpha=alpha, linewidth=linewidth, color3f=colors[i])

def trisurf(vertices, faces, rendertype = 'surface', 
            linewidth = 1.0, alpha = 1.0, color3f = None):
    ''' show trimesh
    
    rendertype = 'mode', 'wireframe', 'surface'
    '''
    vertices = np.atleast_2d(vertices)
    if vertices.shape[-1] == 2:
        vertices = augment_points(vertices)
        
    color3f = get_color_value(color3f)
    argdict = {}
    argdict['representation'] = rendertype
    argdict['line_width'] = linewidth
    argdict['tube_radius'] = linewidth/100
    argdict['opacity'] = alpha
    if color3f is not None:
        argdict['color'] = tuple(color3f)
    mlab.triangular_mesh(vertices[:,0], vertices[:,1], vertices[:,2], faces, **argdict)    
    #if color3f is None:
        #mlab.triangular_mesh(vertices[:,0], vertices[:,1], vertices[:,2], faces,
                             #representation = rendertype, line_width = linewidth,
                             #tube_radius = linewidth, opacity = alpha)
    #else:
        #mlab.triangular_mesh(vertices[:,0], vertices[:,1], vertices[:,2], faces,
                                     #representation = rendertype, line_width = linewidth,
                                     #tube_radius = linewidth, opacity = alpha)         
def show_graph(pts, adjmat, **kwargs):
    if sparse.issparse(adjmat):
        ii, jj, _ = sparse.find(sparse.tril(adjmat))
    else:
        ii, jj = np.nonzero(np.tril(adjmat))
    pts1 = pts[ii]
    pts2 = pts[jj]
    show_segments(pts1,pts2,**kwargs)

def show_segments(pts1,pts2, color3f = None, 
                  alpha = 1.0, linewidth = 1.0):
    assert len(pts1) == len(pts2),'pts1 and pts2 must have equal number of points'
    
    color3f = get_color_value(color3f)
    
    pts1 = vec1n_to_mat1n(pts1)
    pts2 = vec1n_to_mat1n(pts2)
    
    xs = np.concatenate((pts1[:,0].flatten(), pts2[:,0].flatten()))
    ys = np.concatenate((pts1[:,1].flatten(), pts2[:,1].flatten()))
    zs = np.concatenate((pts1[:,2].flatten(), pts2[:,2].flatten()))
    ii = np.arange(len(pts1))
    jj = np.arange(len(pts2)) + len(pts1)
    edges = np.column_stack((ii,jj))
    
    src = mlab.pipeline.scalar_scatter(xs,ys,zs)
    src.mlab_source.dataset.lines = edges
    lines = mlab.pipeline.stripper(src)
    
    argdict = {}
    argdict['opacity'] = alpha
    argdict['line_width'] = linewidth
    if color3f is not None:
        argdict['color'] = tuple(color3f)
    mlab.pipeline.surface(lines, **argdict)
    
def quiver3d(pts, vecs, color3f = None, linewidth = 1.0, alpha=1.0, scale=1.0):
    color3f = get_color_value(color3f)
    pts = vec1n_to_mat1n(pts)
    vecs = vec1n_to_mat1n(vecs)
    
    assert len(pts) == len(vecs)
    argdict = {}
    if color3f is not None:
        argdict['color'] = tuple(color3f)
    if linewidth is not None:
        argdict['line_width'] = linewidth
    if alpha is not None:
        argdict['opacity'] = alpha
    argdict['scale_factor'] = scale
    mlab.quiver3d(pts[:,0], pts[:,1], pts[:,2], vecs[:,0], vecs[:,1], vecs[:,2], **argdict)

#p1 = np.random.rand(10,3)
#p2 = np.random.rand(10,3)
#show_segments(p1,p2, linewidth=10.0)
#show()
def get_color_value(color_str_or_tuple):
    if color_str_or_tuple is None:
        return None
    
    if type(color_str_or_tuple) is str:
        return str2color[color_str_or_tuple]
    else:
        return (color_str_or_tuple[0],color_str_or_tuple[1],color_str_or_tuple[2])