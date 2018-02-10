'''
demo code for loading FLAME face model
Tianye Li <tianye.li@tuebingen.mpg.de>
Based on the hello-world script from SMPL python code
http://smpl.is.tue.mpg.de/downloads
'''

import numpy as np
from os.path import join
from smpl_webuser.serialization import load_model
from fitting.util import write_simple_obj, safe_mkdir

# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # Load FLAME model (here we load the female model)
    # Make sure path is correct
    model_path = './models/female_model.pkl'
    model = load_model( model_path )           # the loaded model object is a 'chumpy' object, check https://github.com/mattloper/chumpy for details
    print "loaded model from:", model_path

    # Show component number
    print "\nFLAME coefficients:"
    print "shape (identity) coefficient shape =", model.betas[0:300].shape # valid shape component range in "betas": 0-299
    print "expression coefficient shape       =", model.betas[300:].shape  # valid expression component range in "betas": 300-399
    print "pose coefficient shape             =", model.pose.shape

    print "\nFLAME model components:"
    print "shape (identity) component shape =", model.shapedirs[:,:,0:300].shape
    print "expression component shape       =", model.shapedirs[:,:,300:].shape
    print "pose corrective blendshape shape =", model.posedirs.shape
    print ""

    # -----------------------------------------------------------------------------
    '''
    safe_mkdir('./output/iterate/')
    value =np.array([-10,-8,-6,-4,-2,0,2,4,6,8,10])
    for i in range(0,1):
        for j in range (value.size):
            model.betas[:] = 0
            model.betas[i] = value[j]
            outmesh_path = './output/iterate/'+'pose_'+str(i)+'='+str(value[j])+'.obj'
            write_simple_obj(mesh_v=model.r, mesh_f=model.f, filepath=outmesh_path)
    '''
    safe_mkdir('./output/iterate_expression/')
    value =np.array([-4,-2,2,4])
    for i in range(300,320):
        for j in range (value.size):
            model.betas[:] = 0
            model.betas[i] = value[j]
            outmesh_path = './output/iterate_expression/'+'express_'+str(i)+'='+str(value[j])+'.obj'
            write_simple_obj(mesh_v=model.r, mesh_f=model.f, filepath=outmesh_path)

    '''
    # Assign random pose and shape parameters
    model.pose[:]  = np.random.randn( model.pose.size ) * 0.05
    model.betas[:] = np.random.randn( model.betas.size ) * 1.0
    # model.trans[:] = np.random.randn( model.trans.size ) * 0.01   # you may also manipulate the translation of mesh

    # Write to an .obj file
    outmesh_dir = './output'
    safe_mkdir( outmesh_dir )
    outmesh_path = join( outmesh_dir, 'hello_flame2.obj' )
    write_simple_obj( mesh_v=model.r, mesh_f=model.f, filepath=outmesh_path )

    # Print message
    print 'output mesh saved to: ', outmesh_path 
    '''
