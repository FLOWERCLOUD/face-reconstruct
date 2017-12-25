import matlab
import matlab.engine as eng
import numpy as np
import atexit
import time
import array

#def get_matlab(just_check = False):
    #has_session = hasattr(get_matlab,'session')
    #if just_check:
        #return has_session
    
    #if not has_session:
        #get_matlab.session = eng.connect_matlab()
        
    #return get_matlab.session

def get_matlab(just_check = False, init_only = False, use_connect = False):
    ''' get matlab session
    
    Parameters
    --------------------
    just_check
        just check whether the session exists, return immediately, will not wait for session ready
        
    init_only
        only initialize the session, will not wait for it to be ready
        
    use_connect
        connect to existing matlab session or create a new session?
        
    Return
    ------------------
    the matlab session
    '''
    has_session = hasattr(get_matlab,'session')
    if just_check:
        return has_session
    
    if not has_session:
        if use_connect:
            print('matlab connected')
            get_matlab.session = eng.connect_matlab(async = True)
        else:
            print('matlab started')
            get_matlab.session = eng.start_matlab(async=True)
        if init_only: #only initialize
            return None
    
    # wait until session is opened
    s = get_matlab.session
    #print('getting matlab session...')
    ml = s.result()
    #print('done')
    return ml

# intialize matlab connection
get_matlab(init_only=True, use_connect=False)

def dtype2arraytype(dtype=None):
    if dtype == np.int64:
        return 'q'
    elif dtype == np.int32:
        return 'l'
    elif dtype == np.int16:
        return 'i'
    elif dtype == np.int8:
        return 'b'
    elif dtype == np.bool:
        return 'B'
    elif dtype == np.uint8:
        return 'B'
    elif dtype == np.uint16:
        return 'I'
    elif dtype == np.uint32:
        return 'L'
    elif dtype == np.uint64:
        return 'Q'
    elif dtype == np.float32:
        return 'f'
    elif dtype == np.float64:
        return 'd'     

def dtype2matlab(dtype=None):
    if dtype == np.int64:
        return matlab.int64
    elif dtype == np.int32:
        return matlab.int32
    elif dtype == np.int16:
        return matlab.int16
    elif dtype == np.int8:
        return matlab.int8
    elif dtype == np.bool:
        return matlab.logical
    elif dtype == np.uint8:
        return matlab.uint8
    elif dtype == np.uint16:
        return matlab.uint16
    elif dtype == np.uint32:
        return matlab.uint32
    elif dtype == np.uint64:
        return matlab.uint64    
    elif dtype == np.float32 or dtype == np.float64:
        return matlab.double    

# type conversion from numpy to matlab
def np2matlab(x, dtype=None):
    if type(x) != np.ndarray:
        x = np.array(x)
        
    dtype = x.dtype
    if dtype == np.float32: #do not support float32
        dtype = np.float64
        
    x = x.astype(dtype = dtype)
    
    # create array
    array_type = dtype2arraytype(dtype=dtype)
    x_arr = array.array(array_type, x.T.flat)    
    
    # create matlab buffer
    matlab_maker = dtype2matlab(dtype=dtype)
    y = matlab_maker(size = x.shape)
    
    # assign to matlab
    y._data[:] = x_arr[:]
    return y

def imtool(img):
    img_ml = np2matlab(img)
    get_matlab().imtool(img_ml)
    
def imshow(img, *args):
    img_ml = np2matlab(img)
    get_matlab().imshow(img_ml, *args)
    
def imwrite(img, filename, *args):
    img_ml = np2matlab(img)
    get_matlab().workspace['tmp']=img_ml
    eval_m('imwrite(tmp,\'{}\');'.format(filename))
    #get_matlab().imwrite(img_ml, filename, *args)
    
def scatter_pts(pts, *args):
    pts = np.atleast_2d(pts)
    pts_ml = np2matlab(pts.T)
    get_matlab().util_scatter_pts(pts_ml, *args)
    
def plot(xs,ys, *args):
    xs = np.array(xs)
    ys = np.array(ys)
    xs_ml = np2matlab(xs)
    ys_ml = np2matlab(ys)
    get_matlab().plot(xs_ml, ys_ml, *args)
    
def mesh_z(z, *args):
    z_ml = np2matlab(z)
    get_matlab().mesh(z_ml, *args)
    
def eval_m(cmdstr, nargout=0):
    ''' evaluate a matlab command
    '''
    get_matlab().eval(cmdstr, nargout=nargout)
    
def setvar(var_name, var_value):
    ''' set a value in matlab's workspace
    '''
    get_matlab().workspace[var_name] = var_value
    
#x = np.random.rand(100,100)>0.5
#imtool(x)