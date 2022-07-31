import chumpy
#chumpy.demo('matrix')
#chumpy.demo('inheritance')
#chumpy.demo('linalg')
#chumpy.demo('show_tree')
#chumpy.demo('scalar')
#chumpy.demo('optimization')
import chumpy as ch
import numpy as np
import chumpy as ch
from os.path import join

from smpl_webuser.serialization import load_model
from fitting.landmarks import load_embedding, landmark_error_3d
from fitting.util import load_binary_pickle, write_simple_obj, safe_mkdir,mat_save
import scipy.io as sio
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.use("qt4agg")
import matplotlib.pyplot
import myplot.vtkplot as vp
import quaternion

x1, x2, x3, x4 = ch.eye(10), ch.array(1), ch.array(5), ch.array(10)
print "model.trans:"
print x1
y = x1*(x2-x3)+x4
print x1,x2,x3,x4
print y
print y.dr_wrt(x2)

def kk():
    hua = 1
    sddd = 2

kk()