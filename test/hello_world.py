'''
demo code for loading FLAME face model
Tianye Li <tianye.li@tuebingen.mpg.de>
Based on the hello-world script from SMPL python code
http://smpl.is.tue.mpg.de/downloads
'''

import numpy as np
from os.path import join
from smpl_webuser.serialization import load_model
from fitting.util import write_simple_obj, safe_mkdirs,FileFilt,read_igl_obj,get_vertex_normal,write_full_obj
import sys
sys.path.insert(0, "D:/mproject/meshlab2016/meshlab/src/x64/Release/")
import meshlab_python


# -----------------------------------------------------------------------------
def render_to_image(input_mesh_dir,out_put_dir):
    safe_mkdirs(out_put_dir)
    b = FileFilt()
    b.FindFile(dirr=input_mesh_dir)

    for k in b.fileList:
        if k == '':
            continue
        filename_split = k.split("/")[-1].split(".")

        if len(filename_split) > 1:
            print str(filename_split[-2])
            file_name = str(filename_split[-2])
        else:
            file_name = str(filename_split[0])
        v_frame_aligned, f_frame_aligned, t_frame_aligned, t_f_frame_aligned, n_frame_aligned, n_f_frame_aligned = read_igl_obj(
            input_mesh_dir + '/'+file_name+'.obj')
        vn_frame_align_to_image = get_vertex_normal(v_frame_aligned, f_frame_aligned)
        v_color_caculate = np.zeros(v_frame_aligned.shape, np.uint8)
        v_color_caculate[:, :] = [255, 255, 255]
        ray_dir = np.array([0, 0, 1])
        ray_dir = np.reshape(ray_dir, (3, 1))
        dot_coef = np.dot(vn_frame_align_to_image, ray_dir)
        dot_coef = np.array([0 if x < 0.01 else x for x in dot_coef])
        dot_coef = np.reshape(dot_coef, (dot_coef.shape[0], 1))
        v_color_caculate = v_color_caculate * dot_coef
        v_color_caculate = v_color_caculate * 0.7 + (1 - 0.7) * np.array([255, 255, 255])
        v_color_caculate = v_color_caculate.astype(np.intc)
        bbox_list = [float(-0.2), float(-0.2), float(-0.2), float(0.2),
                     float(0.2), float(2)]
        # bbox_list = [float(-1.36), float(-1.36), float(-1.3), float(1.36),
        #              float(1.36), float(0.66)]
        # bbox_list = [float(-2.36), float(-2.36), float(-2.36), float(2.36),
        #              float(2.36), float(0.66)]
        result = meshlab_python.Mesh_render_to_image_withmy_bbox(out_put_dir + '/' +
                                                                 file_name + '.png',
                                                                 v_frame_aligned.tolist(), f_frame_aligned.tolist(), [], [], [], [],
                                                                 v_color_caculate.tolist(),
                                                                 int(512), int(512), bbox_list)

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
    # safe_mkdirs('./output/iterate_expression/')
    # value =np.array([-4,-2,2,4])
    # for i in range(300,320):
    #     for j in range (value.size):
    #         model.betas[:] = 0
    #         model.betas[i] = value[j]
    #         outmesh_path = './output/iterate_expression/'+'express_'+str(i)+'='+str(value[j])+'.obj'
    #         write_simple_obj(mesh_v=model.r, mesh_f=model.f, filepath=outmesh_path)

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
    if 1:
        #render_to_image(input_mesh_dir ='D:/mproject/face-reconstruct/output/female_iterate_beta_90/',out_put_dir='D:/mproject/face-reconstruct/output/female_iterate_beta_90/render/')
        #render_to_image(input_mesh_dir ='D:/mproject/face-reconstruct/output/male_iterate_beta_90/',out_put_dir='D:/mproject/face-reconstruct/output/male_iterate_beta_90/render/')
        #render_to_image(input_mesh_dir='D:/mproject/face-reconstruct/output/male_iterate_beta/',
         #               out_put_dir='D:/mproject/face-reconstruct/output/male_iterate_beta')
        if 0:
            render_to_image(input_mesh_dir='D:/mproject/face-reconstruct/output/iterate_expression/',
                            out_put_dir='D:/mproject/face-reconstruct/output/iterate_expression/render/')
        render_to_image(input_mesh_dir='D:\mproject/face-reconstruct\output/flame_blendshape/test_transfer/fast_transfer/',
                        out_put_dir='D:\mproject/face-reconstruct\output/flame_blendshape/test_transfer/fast_transfer/render/')

    if 0:
        render_to_image(input_mesh_dir='E:\workspace/vrn_data\Tester_104\Bleandshape_triangle/',
                        out_put_dir='E:\workspace/vrn_data\Tester_104\Bleandshape_triangle/render/')
