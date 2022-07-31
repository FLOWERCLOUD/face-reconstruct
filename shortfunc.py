import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as slg
#from .geometry import adjmat_from_mesh, laplacian_from_adjmat
#from .inout import imread, imwrite

def random_string(n):
    import uuid
    return str(uuid.uuid4())
    #import string
    #charlist = list(string.ascii_letters + string.digits)
    #x = np.random.choice(charlist, n).tostring()
    #return x

def apply_colormap_to_labelmap(x, label2color = None):
    ''' convert label map x into a color image based on a colormap
    
    label2color[i] = (r,g,b) for label i, (r,g,b) are [0,255] integer values
    
    x = a label image, where labels are integers
    
    return x_rgb, a MxNx3 color image
    '''
    assert len(x.shape)==2, 'only accept 2d label map'
    if label2color is None:
        lbs = np.unique(x.flat)
        colors = np.random.rand(max(lbs)+1,3)
        colors[0] = (0,0,0)
        colors *= 255
        colors = colors.astype('uint8')
        label2color = colors
    
    lb2r = label2color[:,0]
    lb2g = label2color[:,1]
    lb2b = label2color[:,2]
    
    img_r = lb2r[x]
    img_g = lb2g[x]
    img_b = lb2b[x]
    
    output = np.dstack((img_r, img_g, img_b))
    return output
    
def mat2gray(x):
    x = np.array(x).astype(float)
    x_min = x.min()
    x_max = x.max()
    if x_max == x_min:
        if x_max == 0:
            y=x
        else:
            y=x/x_max
    else:
        y = (x-x_min)/(x_max - x_min)
    return y

def find_contiguous_interval(mask1d):
    ''' given a binary mask, e,g.[0,0,0,1,1,1,0,1,1,1,0,...],
    find end points of contiguous 1s. For example, 
    for p=[0,0,1,1,1,0], return [2,5), meaning that p[2:5] is a
    series of 1s
    
    return Nx2 endpts, where endpts[i]=[a,b), which means mask1d[a:b]
    is a contiguous region of 1s
    '''
    mask1d = np.array(mask1d).flatten() > 0
    tmp = np.zeros(len(mask1d)+2, dtype=int)
    tmp[1:-1] = mask1d
    vals = tmp[1:] - tmp[:-1]
    v_begin = np.nonzero(vals>0)[0]
    v_end = np.nonzero(vals<0)[0]
    return np.column_stack((v_begin, v_end))
    
    
def isMember(a,b):
    '''determine whether a[i] is in b,
    
    return (tf,locb),
    
    where tf[i]=True iff a[i] is in b, and a[i]=b[locb[i]],
    
    locb[i]=-1 for each tf[i]=False
    '''
    if type(a) is np.ndarray and len(a.shape)>1:
        a = a.tolist()
        a = [tuple(x) for x in a]
    if type(b) is np.ndarray and len(a.shape)>1:
        b = b.tolist()
        b = [tuple(x) for x in b]
        
    # create a dictionary using b's values
    bdict = dict()
    for i, val in enumerate(b):
        if val not in bdict:
            bdict[val]=i
    locb = [bdict.get(x, -1) for x in a]
    tf = [x>=0 for x in locb]
    
    tf = np.array(tf)
    locb = np.array(locb, dtype=int)
    return (tf,locb)

# rename the function
is_member = isMember

def mldivide(a,b):
    ''' inv(a).dot(b), equivalent to matlab's a\b, will handle sparse matrix.
    '''
    if sparse.issparse(a):
        if a.shape[0] == a.shape[1]:
            output = slg.spsolve(a,b)
        else:
            ata = a.transpose().dot(a).tocsc()
            atb = a.transpose().dot(b)
            output = slg.spsolve(ata,atb,use_umfpack=False)
            #lu = slg.splu(ata)
            #output = lu.solve(atb)
    else:
        output = np.linalg.lstsq(a,b)[0]
    return output

def mrdivide(a,b):
    ''' a.dot(inv(b)), equivalent to matlab's a/b
    '''
    x = np.linalg.lstsq(b.T,a.T)[0].T
    return x

def unique_rows(x, stable = False):
    ''' find unique rows in x
    
    parameters
    -------------------
    x
        nxd matrix, each row is a data entry
    stable
        should we return the unique rows in the same order as they appear in x?
    
    return
    -------------------------
    out_rows
        unique rows in x
    index
        indices of the unique rows in x, such that x[index]==out_rows
    inv_index
        indices for reconstructing x from out_rows, such that x==out_rows[inv_index]
    count
        count[i] is the number of times unique_rows[i] appear in x
    '''
    if len(x.shape) == 1:
        _, idx, idxinv, count = np.unique(x,return_index=True, return_inverse=True, return_counts=True)
    else:
        b = np.ascontiguousarray(x).view(np.dtype((np.void, x.dtype.itemsize * x.shape[1])))
        _, idx, idxinv, count = np.unique(b, return_index=True, return_inverse=True, return_counts=True)
    if stable:
        ii = np.argsort(idx)
        idx = idx[ii]
        count = count[ii]
        
        idxold2new = np.zeros(len(idx),dtype=idxinv.dtype)
        idxold2new[ii] = np.arange(len(idx),dtype=idxinv.dtype)
        idxinv = idxold2new[idxinv]
        
        #idx = np.sort(idx)
    output = x[idx]
    return (output, idx, idxinv, count)

def find_row(x, rowval):
    ''' find rows in x which are equal to rowval
    
    return idx such that x[idx[i]]==rowval
    '''
    assert len(x.shape)==2, 'x should be 2d'
    rowval = np.array(rowval).flatten()
    return np.where((x==rowval).all(axis=1))[0]

def normalize_rows(x):
    ''' normalize the row vectors in x, or 1d vector
    
    return (x, lens), where x is the normalized x, and lens[i] is the norm of x[i]
    '''
    if len(x.shape)==1:
        lens = np.linalg.norm(x)
        x = x/lens
        lens = np.array([lens])
    else:
        lens = np.linalg.norm(x,axis=1)
        x=x/lens.reshape((-1,1))
    return (x,lens)

def nonzeros(x):
    ''' return indices of the non zero values in x, x can be sparse or dense
    
    return (ii,jj,uu,....) where each ii(k),jj(k),uu(k),... locates a nonzero element in x
    '''
    if sparse.issparse(x):
        res = sparse.find(x)
        output = (res[0],res[1])
    else:
        output = np.nonzero(x)
    return output

def sortrows(x):
    ''' sort x treating each row of x as an entry
    
    return (x_sorted, idx), where x_sorted = x[idx]
    '''
    if x.size == 0:
        return (x,np.array([]))
    
    assert len(x.shape)==2, 'can only sort 2d array'
    idx = np.lexsort(x[:,::-1].T)
    x_sorted = x[idx]
    return (x_sorted, idx)

def replace_values(x, dict_old2new):
    ''' replace values in x, such that, y[i] = dict_old2new[x[i]]
    '''
    uvals, index = np.unique(x, return_inverse=True) #uvals[index] == x
    newvals = np.array(uvals)
    for i, v in enumerate(uvals):
        if v in dict_old2new:
            newvals[i] = dict_old2new[v]
            
    y = newvals[index].reshape(x.shape)
    return y
    
def cart_product(x,y):
    ''' return cartisian product of x and y, as nx2 numpy array where output[k]=(x[i],y[j])
    '''
    xx, yy =np.meshgrid(x,y)
    return np.dstack((xx.T,yy.T)).reshape((-1,2))

def vec1n_to_mat1n(x):
    ''' convert 1d numpy array to 1xn matrix if neccessary
    '''
    if x is None:
        return None
    
    if len(x.shape)==1:
        return x.reshape((1,-1))
    else:
        return x
    
def transform_points(pts, transmat):
    pts = vec1n_to_mat1n(pts)
    ptsnew = np.column_stack((pts, np.ones(len(pts))))
    ptsnew = ptsnew.dot(transmat)
    #ptsnew = ptsnew/ptsnew[:,-1:]
    return ptsnew[:,:-1]

def transform_vectors(vecs,transmat):
    vecs = vec1n_to_mat1n(vecs)
    vecsnew = np.column_stack((vecs, np.zeros(len(vecs))))
    vecsnew = mrdivide(vecsnew, transmat.T)
    return vecsnew[:,:-1]

def mkdir(dirname):
    import os
    if not os.path.exists(dirname):
        os.makedirs(dirname)
        #os.mkdir(dirname)
    
def convert_crlf_to_lf(filename):
    #with open(filename) as template:
        #lines = [line.replace('\r\n', '\n') for line in template]
    #with open(filename, 'w') as template:
        #template.writelines(lines)    
    with open(filename, 'rb') as fid:
        data = fid.read()
    
    newdata = data.replace("\r\n", "\n")
    with open(filename, 'wb') as fid:
        fid.write(newdata)
        
def mat2gray(x, to_range = (0,1)):
    ''' normalize x to specific range
    
    to_range = (min,max) which is the target range
    
    return the normalized input
    '''
    y = np.array(x, dtype=float)
    xmin, xmax = x.min(), x.max()
    if xmin == xmax:
        y = np.ones(y.shape) * (to_range[1] - to_range[0]) + to_range[0]
    else:
        y = (x-xmin)/(xmax-xmin)*(to_range[1]-to_range[0]) + to_range[0]
    return y

def image_gradient(img, ksize=3, normalize=True, dx=1, dy=1):
    ''' find image gradient
    
    Parameters
    -----------------
    img
        the image
    ksize
        gradient kernel size
    normalize
        is the gradient kernel normalized or not
    dx, dy
        the order along x (horizontal) and y (vertical) direction
        
    Return
    ----------------
    dx,dy
        the gradient along x and y direction
    '''
    import cv2
    res = cv2.getDerivKernels(dx=dx,dy=0,ksize=ksize,
                              normalize=normalize)
    dx = cv2.sepFilter2D(img,-1,res[0],res[1])
    res = cv2.getDerivKernels(dx=0,dy=dy,ksize=ksize,
                              normalize=normalize)
    dy = cv2.sepFilter2D(img,-1,res[0],res[1])
    return (dx,dy)

def im2row(img, patch_size_hw, step = 1):
    ''' like matlab's im2col. Slide a window across the image,
    and concatenate values of each patch into a row
    
    Parameters
    ------------
    img
        2d or 3d image
    patch_size_hw
        patch size [height, width]
    step
        strides of the sliding window
        
    Return
    ------------
    imgrows
        imgrows[i] = concatenated pixel values of i-th window
    '''
    import skimage.util as skutil
    if len(img.shape) == 2:
        psize = np.array(patch_size_hw)
    else:
        psize = np.append(patch_size_hw, img.shape[-1])
        
    x = skutil.view_as_windows(img, psize, step = step)
    npix = np.prod(psize)
    output = x.reshape((-1, npix))
    return output

def im2wins(img, patch_size_hw, step = 1, return_uv2ij = False):
    ''' slide a window across image, creating a stack of images each
    slice being a window of the original image
    
    Parameters
    -------------
    img
        the image, 2d or 3d
    patch_size_hw
        patch size given as [height, width]
    step
        stride of the sliding window, can be a number or an array
        specifying the step along each dimension
    return_uv2ij
        if True, return two arrays (u2i, v2j), which converts window coordinate
        (u,v) into global pixel coordinate (i,j). Window's anchor is its upper left pixel.
    
    Return
    -------------
    wins
        wins[u,v,i,j] is the pixel (i,j) inside window (u,v)
    u2i, v2j
        the upper left pixel of window (u,v) is at (u2i[u],v2j[v])
    '''
    import skimage.util as skutil
    if len(img.shape) == 2:
        psize = np.array(patch_size_hw)
    else:
        psize = np.append(patch_size_hw, img.shape[-1])
        
    _step = np.ones(len(img.shape), dtype=int)
    if np.isscalar(step):
        _step[:] = step
    else:
        _step[:len(step)] = step
    
    wins = skutil.view_as_windows(img, psize, _step)
    u_step = _step[0]
    v_step = _step[1]    
    u2i = np.arange(wins.shape[0]) * u_step
    v2j = np.arange(wins.shape[1]) * v_step
    if return_uv2ij:
        return wins, u2i,v2j
    else:
        return wins

def fileparts(filename, ext_by_first_dot = False):
    ''' split file name into path, filename and extension
    
    parameters
    -------------------------
    filename
        the filename to be splitted
    ext_by_first_dot
        if there are multiple dots in the filename, e.g.,
        'abcd.efg.hi.jk', which part should be considered as extension?
        If True, then the extension begins from the first dot, 
        otherwise it begins from the last dot.
    
    return
    -----------------------
    path
        file path
    name
        file name without extension
    extension
        extention with '.'
    '''
    import os
    filename = os.path.normpath(filename)
    path, name_ext = os.path.split(filename)
    if ext_by_first_dot:
        idx = name_ext.find('.')
        if idx==-1:
            name = name_ext
            ext = ''
        else:
            name = name_ext[:idx]
            ext = name_ext[idx:]
    else:
        name, ext = os.path.splitext(name_ext)
    return (path, name, ext)

def print_object(obj):
    import pprint
    pprint.pprint(vars(obj))
    
# adjmat may be sparse
def adjmat_to_laplacian(adjmat):
    ''' create laplacian matrix from adjmat
    
    if adjmat is sparse, return sparse matrix, otherwise return full matrix
    '''
    import scipy.sparse as sparse
    adjmat = adjmat.astype('float')
    dgs = adjmat.sum(axis=1)
    if sparse.issparse(adjmat):
        inv_dgs = sparse.csr_matrix(1/dgs)
        L = adjmat.multiply(inv_dgs) - sparse.eye(adjmat.shape[0])
    else:
        L = adjmat/dgs - np.eye(len(adjmat))
    return L


# adjmat may be sparse
def adjmat_to_laplacian2(adjmat):
    ''' create laplacian matrix from adjmat

    if adjmat is sparse, return sparse matrix, otherwise return full matrix
    '''
    import scipy.sparse as sparse
    adjmat = adjmat.astype('float')
    dgs = adjmat.sum(axis=1)
    if sparse.issparse(adjmat):
        inv_dgs = sparse.csr_matrix(1 / dgs)
        L =  sparse.eye(adjmat.shape[0]) - adjmat.multiply(inv_dgs)
    else:
        L = np.eye(len(adjmat)) -adjmat / dgs
    return L

def find_transform_projective(pts_1, pts_2):
    ''' find a projective transformation that transforms pts_1 to pts_2
    
    return transmat, a 4x4 matrix for 3d points, 3x3 matrix for 2d points
    '''
    p1 = np.column_stack((pts_1, np.ones(len(pts_1))))
    p2 = np.column_stack((pts_2, np.ones(len(pts_2))))
    transmat = mldivide(p1,p2)
    return transmat

def find_transform_similarity(pts_1, pts_2):
    ''' find a similarity transformation that maps pts_1 to pts_2, which means
    pts_2 = pts_1.dot(R)*s+t.
    
    The method is described in:
    S. Umeyama, "Least-squares estimation of transformation parameters between two point patterns," 
    in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 13, no. 4, pp. 376-380, Apr 1991.
    
    return transmat, 4x4 matrix for 3d points and 3x3 for 2d points
    '''
    pts_1 = np.atleast_2d(pts_1)
    pts_2 = np.atleast_2d(pts_2)
    
    ndim = pts_1.shape[-1]
    c1 = pts_1.mean(axis=0)
    p1 = pts_1 - c1
    var1= (p1**2).sum(axis=1).mean()
    
    c2 = pts_2.mean(axis=0)
    p2 = pts_2 - c2
    var2 = (p2**2).sum(axis=1).mean()
    
    #sigmat = 0
    #for i in xrange(len(p1)):
        #sigmat += np.outer(p2[i], p1[i])
    #sigmat /= len(p1)
    sigmat = np.einsum('ik,jk->ij', p2.T, p1.T)/len(p1)
    U,D,Vt = np.linalg.svd(sigmat)
    D = np.diag(D)
    
    S = np.eye(sigmat.shape[0])
    sigrank = np.linalg.matrix_rank(sigmat)
    if sigrank == ndim:
        #full rank
        if np.linalg.det(sigmat)<0:
            S[-1,-1] = -1
    elif sigrank == ndim-1:
        dval = np.linalg.det(U) * np.linalg.det(Vt.T)
        if dval < 0:
            S[-1,-1] = -1
    else:
        assert False, 'rank is less than ndim-1, similarity transformation cannot be found'
    
    # create left-multiply transformation matrix
    rotmat = U.dot(S).dot(Vt)
    scale = 1/var1 * np.trace(D.dot(S))
    tranvec = c2[:,np.newaxis] - scale * rotmat.dot(c1[:,np.newaxis])
    transmat = np.eye(4)
    transmat[:3,:3] = scale * rotmat
    transmat[:3,-1] = tranvec.flatten()
    
    # convert to right multiply format
    return transmat.T

def rotation_matrix_by_angles(x_rad, y_rad, z_rad):
    ''' create rotation matrix by rotating around x,y,z axis
    '''
    x_cos, y_cos, z_cos = np.cos([x_rad, y_rad, z_rad])
    x_sin, y_sin, z_sin = np.sin([x_rad, y_rad, z_rad])
    Rx = np.array([[1,0,0],[0,x_cos,-x_sin],[0,x_sin,x_cos]]).T
    Ry = np.array([[y_cos,0,y_sin],[0,1,0],[-y_sin,0,y_cos]]).T
    Rz = np.array([[z_cos,-z_sin,0],[z_sin,z_cos,0],[0,0,1]]).T
    rotmat = Rx.dot(Ry).dot(Rz)
    tmat = np.eye(4)
    tmat[:-1,:-1] = rotmat
    return tmat

def translation_matrix(dx,dy,dz):
    transmat = np.eye(4)
    transmat[-1,:-1]=[dx,dy,dz]
    return transmat
    
def scale_matrix(sx,sy = None,sz = None):
    if sy is None and sz is None:
        sy = sx
        sz = sx
    tmat = np.eye(4)
    tmat[:-1,:-1] = np.diag([sx,sy,sz])
    return tmat

def transformation_matrix(trans_vec, rot_angles_rad, scale_vec):
    R = rotation_matrix_by_angles(*rot_angles_rad)
    S = scale_matrix(*scale_vec)
    T = translation_matrix(*trans_vec)
    final = S.dot(R).dot(T)
    return final

def get_colormap(name, n):
    ''' get colormap with n colors, colors are floats in range [0,1]
    
    name = 'hot','hsv','jet'
    
    n = number of colors
    '''
    import matplotlib.cm as cm
    pre_colors = None
    if name == 'hsv':
        cm.hsv(0)
        pre_colors = cm.hsv._lut
    elif name == 'hot':
        cm.hot(0)
        pre_colors = cm.hot._lut
    elif name == 'jet':
        cm.jet(0)
        pre_colors = cm.jet._lut
    else:
        assert False,'colormap not recognized'
        
    pre_colors = pre_colors[:-1] #the last entry is 0,0,0,0 which should be ignored
    import scipy.interpolate as itp
    xs = np.linspace(0,1,len(pre_colors))
    obj = itp.interp1d(xs, pre_colors,axis=0)
    ts = np.linspace(0,1,n)
    return obj(ts)

def get_type_name(obj):
    ''' get the fully qualified type name of obj
    '''
    x = '{0}'.format(type(obj))
    import re
    ptn = r"<class '(?P<module_name>.+)'>"
    res = re.search(ptn,x)
    typename = res.groupdict()['module_name']
    return typename

def is_tensorflow_object(obj):
    ''' determine if obj is a tensorflow object.
    '''
    cname = get_type_name(obj)
    if cname.find('tensorflow') == 0:
        return True
    else:
        return False

def is_theano_object(obj):
    ''' determine if obj is a theano object
    '''
    
class ObjectWithPPrint:
    def __repr__(self):
        import pprint
        s = pprint.pformat(vars(self))
        return s
    
class Unsettable:
    def unset(self):
        ''' set all fields to None
        '''
        for key, val in vars(self).items():
            setattr(self, key, None)
        

class CallbackParam(ObjectWithPPrint):
    def __init__(self):
        # if all(abs(x1-x0)<thres), optimization terminates
        # x1 and x0 are variables in consecutive iteration
        self.thres_dx = None
        
        # the above threshold applies every n iteration
        # rather than consecutive iteration
        self.evaluate_dx_per_n_iter = 1
        
        # if abs(cost1 - cost0)<thres, optimization terminates
        # cost1 and cost0 are objective values in consecutive iteration
        self.thres_objective_value = None
        
        #the above threshold applies to objective values every n iterations
        #rather than consecutive iteration
        self.evaluate_objective_diff_per_n_iter = 1
        
        # print objective value per n iteration
        # set to None to disable printing
        self.print_per_n_iteration = 10
        
        # stop after this many seconds
        # set to None or np.inf to mean infinity
        self.time_limit = None

def make_scipy_optimization_callback(outdict, objfunc, cbparam):
    ''' create a callback used in scipy.optimization.minimize
    
    parameters
    ----------------------
    outdict
        a dict() for writing output
    objfunc
        objective function
    cbparam
        the callback param
        
    output
    ---------------------
    callback
        a function f(x) that can be used as callback. 
        When the callback wants to terminate the optimization,
        a ValueError exception is raised
    '''
    
    import time
    if False:
        cbparam = CallbackParam()
    
    def callback(x):
        clock_begin = callback.clock_begin
        ith_iter = callback.ith_iter
        x_prev = callback.x_prev
        y_prev = callback.y_prev
        
        outdict['x'] = x
        objval = objfunc(x)
        outdict['y'] = objval        
        
        # check time
        t = time.clock()
        if clock_begin is None:
            callback.clock_begin = t
            time_elapsed = 0
        else:
            time_elapsed = t - clock_begin
            if time_elapsed > cbparam.time_limit:
                raise ValueError('time limit reached')  
        
        # check dx
        if ith_iter % cbparam.evaluate_dx_per_n_iter == 0:
            if callback.x_prev is None:
                callback.x_prev = np.array(x)
            else:
                dx = np.abs(x - x_prev)
                if cbparam.thres_dx is not None:
                    if np.all(dx < cbparam.thres_dx):
                        raise ValueError('terminates by all(abs(dx)) < %s' % cbparam.thres_dx)
                callback.x_prev[:] = x[:]
            
        # check dy
        if ith_iter % cbparam.evaluate_objective_diff_per_n_iter == 0:          
            if callback.y_prev is None:
                callback.y_prev = objval
            else:
                dy = np.abs(objval - y_prev)
                if cbparam.thres_objective_value is not None:
                    if dy < cbparam.thres_objective_value:
                        raise ValueError('terminates by dy < %s' % cbparam.thres_objective_value)
            callback.y_prev = objval
            
        # print
        if ith_iter % cbparam.print_per_n_iteration == 0:
            print('%d(%.2f s): %f' % (ith_iter, time_elapsed, objval))
        callback.ith_iter += 1
            
    callback.clock_begin = None
    callback.ith_iter = 0
    callback.x_prev = None
    callback.y_prev = None
    return callback
    
def pca_by_svd(pts):
    ''' compute pca basis and coefficients using svd
    
    parameters
    --------------------
    pts
        the points in nxd format
        
    return
    --------------------
    pca_basis
        dxd basis, each row is a direction, the first is the max
    pca_coef
        the coefficient for each basis
    '''
    p = pts - pts.mean(axis=0)
    u,s,v = np.linalg.svd(p.T.dot(p)/len(pts))
    return v, s

def make_frame(dir3):
    ''' generate an arbitrary orthogonal frame given a 3d direction.
    
    parameters
    --------------------
    dir3
        a given direction
        
    return
    ----------------------
    frame
        3x3 matrix, (x,y,z), each row is a direction, z is the given direction dir3.
    '''
    t = np.atleast_1d(dir3).flatten()
    t = t/np.linalg.norm(t)
    idxmax = np.argmax(np.abs(t))
    if idxmax == 0:
        t[0], t[1] = t[1], -t[0]
    elif idxmax == 1:
        t[0], t[1] = -t[1], t[0]
    else:
        t[0], t[2] = -t[2], t[0]
        
    x = np.array(dir3)
    y = np.cross(x,t)
    z = np.cross(x,y)
    
    x = x/np.linalg.norm(x)
    y = y/np.linalg.norm(y)
    z = z/np.linalg.norm(z)
    
    return np.row_stack((y,z,x))

def copy_fe_symlink(dir_fe_in, dir_out):
    '''  "Privilege not held" error
    To change the policies:
    --------------------
    1.Launch secpol.msc via Start or Start -> Run.
    2.Open Security Settings ->  Local Policies ->  User Rights Assignment.
    3.In the list, find the "Create symbolic links" item, which represents SeCreateSymbolicLinkPrivilege.
    4.Double-click on the item and add yourself (or the whole Users group) to the list.
    
    --------------------
    The changes will apply when you log out and log in again.
      
    '''   
    
    import os      
    dir_out_fe = dir_out + '\\face_exchange'
    mkdir(dir_out_fe) 

    dir_temp_output = dir_out_fe + '\\temp'
    mkdir(dir_temp_output)    

    listdir = os.listdir(dir_fe_in)
    for fn in listdir:
        if fn != 'temp' :
            src = dir_fe_in + '\\' + fn
            dst = dir_out_fe + '\\' + fn
            if not os.path.exists(dst):
                os.symlink(src, dst)           

def assign_color_to_image(img, ij_or_mask, colors, return_copy = True):
    ''' assign colors to locations of the image, handling channel difference
    
    parameters
    ---------------------
    img
        the image to be written to
    ij_or_mask
        image positions to write, 
        can be nx2 integer matrix representing ij coordinates,
        or a mask the same size as the image to indicate where to write
    colors
        nx3 RGB, or nx4 RGBA, or 1xn grayscale colors to be written
        into the image. 
        Must have len(colors) == len(ij)
        or len(colors) == mask.sum()
    return_copy
        should we return a copy of the img, or modify img directly
        
    return
    ------------------------
    imgout
        an image with imgout[ij_or_mask] == colors
    '''
    ij_or_mask = np.atleast_2d(ij_or_mask)
    if ij_or_mask.dtype == np.bool:
        ii, jj = np.nonzero(ij_or_mask)
    else:
        ii, jj = ij_or_mask.T
    assert len(ii) == len(colors),'number of colors and number of locations do not match'
    
    if return_copy:
        imgout = np.array(img)
    else:
        imgout = img
        
    if len(img.shape) > 2:
        nch = img.shape[-1]
        colors_aug = np.zeros((len(ii), nch), dtype = imgout.dtype)
        nch_use = min(nch, colors.shape[-1])
        colors_aug[:,:nch_use] = colors[:,:nch_use]
        imgout[ii,jj] = colors_aug
    else:
        imgout[ii,jj] = colors[:,0]
        
    return imgout
        
class SparseSolverDecomp:
    ''' For solving Ax=b, stores LU decomposition of sparse system for later use
    '''
    def __init__(self):
        self.L = None
        self.U = None
        self.Pr = None
        self.Pc = None
        self.A = None
        
    @staticmethod
    def init_with_splu(A, splu_obj):
        ''' initialize with the returned object from scipy.sparse.linalg.splu(),
        and the A in Ax=b
        '''
        self = SparseSolverDecomp()
        self.A = A.tocsr()
        n = A.shape[1]
        
        solver = splu_obj
        self.L = solver.L
        self.U = solver.U
    
        ii = solver.perm_r
        jj = np.arange(n)
        data = np.ones(n)
        self.Pr = sparse.csr_matrix((data,(ii,jj)), (n,n))  
    
        ii = np.arange(n)
        jj = solver.perm_c
        data = np.ones(n)
        self.Pc = sparse.csr_matrix((data,(ii,jj)), (n,n))        
        
        return self
    
    @staticmethod
    def init_with_sp_matrix(A):
        ''' initialize with the system matrix in Ax=b
        '''
        import scipy.sparse as sparse
        import scipy.sparse.linalg as slg
        
        ata = A.transpose().dot(A).tocsc()
        solver = slg.splu(ata)
        return SparseSolverDecomp.init_with_splu(A, solver)
        
    def solve(self, b):
        ''' solve Ax=b for b, see SuperLU documentation for explanation
        
        return
        -------------
        sol
            the solution of Ax=b
        '''
        import scipy.sparse as sparse
        import scipy.sparse.linalg as slg        
        A,L,U,Pr,Pc = self.A, self.L, self.U, self.Pr, self.Pc
        
        # copy for overwrite
        L = L.tocsr()
        U = U.tocsr()
        b = np.array(b)
        
        atb = A.transpose().dot(b)
        Prb = Pr.dot(atb)
        Uy = slg.spsolve_triangular(L, Prb, lower=True, overwrite_A=True, overwrite_b=True)
        y = slg.spsolve_triangular(U, Uy, lower=False, overwrite_A=True, overwrite_b=True)
        sol = Pc.dot(y)
        return sol
    
def delete_contents_in_dir(dirpath, delete_subdir = True):
    ''' delete all contents in this dir
    
    parameters
    -----------------------
    dirpath
        the path to the directory
    delete_subdir
        also delete sub directories
    '''
    import os, shutil
    folder = dirpath
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif delete_subdir and os.path.isdir(file_path): 
                shutil.rmtree(file_path)
        except Exception as e:
            print(e)