# responsible for plotting image-related things
import cv2
import numpy as np
import common.shortfunc as sf

def imshow(img, winname = None, allow_resize = True):
    if winname is None:
        winname = sf.random_string(10)
        
    if img.dtype == np.bool:
        img = img.astype('uint8') * 255
        
    if allow_resize:
        cv2.namedWindow(winname, flags = cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    else:
        cv2.namedWindow(winname)
        
    # convert rgb into bgr for opencv display
    if len(img.shape)==3:
        img = img[:,:,::-1]
        
    cv2.imshow(winname, img)
    kval = cv2.waitKey(0)
    cv2.destroyAllWindows()