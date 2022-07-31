# -- coding: utf-8 --
import numpy as np
import cv2
from fitting.util import FileFilt
from  bayesian_matting_master.bayesian_matting import matting
from numba import jit,jitclass,int32, float32,float64

def convert_ppm_2_png(inpudir,outputdir):
    b = FileFilt()
    b.FindFile(dirr=inpudir)
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
            file_name =  str(filename_split[0])
        ppm_img = cv2.imread(k, cv2.IMREAD_COLOR)
        output_file = outputdir+file_name+'.png'
        cv2.imwrite(output_file,ppm_img)

def convert_png_2_jpg(inpudir,outputdir):
    b = FileFilt()
    b.FindFile(dirr=inpudir)
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
            file_name =  str(filename_split[0])
        png_img = cv2.imread(k, cv2.IMREAD_COLOR)
        output_file = outputdir+file_name+'.jpg'
        cv2.imwrite(output_file,png_img,[int(cv2.IMWRITE_JPEG_QUALITY), 10])

def convert_jpg_2_png(inpudir,outputdir):
    b = FileFilt()
    b.FindFile(dirr=inpudir)
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
            file_name =  str(filename_split[0])
        jpg_img = cv2.imread(k, cv2.IMREAD_COLOR)
        output_file = outputdir+file_name+'.png'
        #cv2.imwrite(output_file,jpg_img,[int(cv2.IMWRITE_PNG_COMPRESSION), 3])
        cv2.imwrite(output_file, jpg_img)


@jit(nopython=True, cache=True)
def if_equal(color1, color2):
    if color1[0] == color2[0] and color1[1] == color2[1] and color1[2] == color2[2]:
        return True
    else:
        return False
@jit(nopython=True, cache=True)
def convert2(jpg_img,color_array):
    convert_img = np.zeros((jpg_img.shape[0], jpg_img.shape[1], jpg_img.shape[2]), dtype=np.uint8)

    for i in range(0, jpg_img.shape[0]):
        for j in range(0, jpg_img.shape[1]):
            min_dis = 10000000
            color = jpg_img[i, j, :]
            for k in range(0, 4):
                cur_dis = abs(color[0]-color_array[k,0]) +abs(color[1]-color_array[k,1])+abs(color[2]-color_array[k,2])  #np.linalg.norm(color - color_array[k])
                if (cur_dis < min_dis):
                    min_dis = cur_dis
                    index = k
            if index == 0:
                convert_img[i, j, :] = color_array[3]
            else:
                convert_img[i, j, :] = color_array[index]



    for i in range(0, convert_img.shape[0]):
        for j in range(0, convert_img.shape[1]):
            color = convert_img[i, j, :]
            if if_equal(color, color_array[3, :]):
                if i > 0 and i < convert_img.shape[0] - 1 and j > 0 and j < convert_img.shape[1] - 1:
                    color_left = convert_img[i, j - 1, :]
                    color_right = convert_img[i, j + 1, :]
                    color_top = convert_img[i - 1, j, :]
                    color_down = convert_img[i + 1, j + 1, :]
                    if if_equal(color_left, color_array[3, :]):
                        continue
                    if if_equal(color_right, color_array[3, :]):
                        continue
                    if if_equal(color_top, color_array[3, :]):
                        continue
                    if if_equal(color_down, color_array[3, :]):
                        continue
                    convert_img[i, j, :] = color_right
    return convert_img
def convert_jpg_2_png_with_color(inpudir,outputdir):
    import numpy as np

    b = FileFilt()
    b.FindFile(dirr=inpudir)
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
            file_name =  str(filename_split[0])
        jpg_img = cv2.imread(k, cv2.IMREAD_COLOR)
        color_array = np.array([[0, 0, 0],  # 黑色
                                [0, 0, 255],  # 红色
                                [0, 255, 0],  # 绿色
                                [255, 0, 0]])  # 蓝色
        convert_img = convert2(jpg_img,color_array)

        output_file = outputdir+file_name+'.png'
        cv2.imwrite(output_file,convert_img)

@jit(nopython=True, cache=True)
def convert_png2_1channel_wrapper(jpg_img,color_array):
    convert_img = np.zeros((jpg_img.shape[0], jpg_img.shape[1], 1), dtype=np.uint8)

    outpucolor_array = np.array([1, 2, 0, 1])
    for i in range(0, jpg_img.shape[0]):
        for j in range(0, jpg_img.shape[1]):
            min_dis = 10000000
            color = jpg_img[i, j, :]
            for k in range(0, 4):
                cur_dis = abs(color[0]-color_array[k,0]) +abs(color[1]-color_array[k,1])+abs(color[2]-color_array[k,2])#np.linalg.norm(color - color_array[k])
                if (cur_dis < min_dis):
                    min_dis = cur_dis
                    index = k
            if index == 0:
                convert_img[i, j, :] = outpucolor_array[3]
            else:
                convert_img[i, j, :] = outpucolor_array[index]
    return convert_img
def convert_png2_1channel(inpudir,outputdir):
    import numpy as np

    b = FileFilt()
    b.FindFile(dirr=inpudir)
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
            file_name =  str(filename_split[0])
        jpg_img = cv2.imread(k, cv2.IMREAD_COLOR)
        color_array = np.array([[0, 0, 0],  # 黑色
                                [0, 0, 255],  # 红色
                                [0, 255, 0],  # 绿色
                                [255, 0, 0]])  # 蓝色
        convert_img = convert_png2_1channel_wrapper(jpg_img,color_array)
        output_file = outputdir+file_name+'.png'
        cv2.imwrite(output_file,convert_img)

@jit(nopython=True, cache=True)
def convert_1channel_2png_wrapper(jpg_img,color_array,outpucolor_array):
    convert_img = np.zeros((jpg_img.shape[0], jpg_img.shape[1], 3), dtype=np.uint8)


    for i in range(0, jpg_img.shape[0]):
        for j in range(0, jpg_img.shape[1]):
            color = jpg_img[i, j]
            if color == color_array[0]:
                convert_img[i, j] = outpucolor_array[0]
            elif color == color_array[1]:
                convert_img[i, j] = outpucolor_array[1]
            elif color == color_array[2]:
                convert_img[i, j] = outpucolor_array[2]
            elif color == color_array[3]:
                convert_img[i, j] = outpucolor_array[3]
            else:
                convert_img[i, j] = [255,255,255]
    return convert_img

def convert_1channel_png2(inpudir,outputdir):
    import numpy as np

    b = FileFilt()
    b.FindFile(dirr=inpudir)
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
            file_name =  str(filename_split[0])
        jpg_img = cv2.imread(k, cv2.IMREAD_GRAYSCALE)
        color_array = np.array([3,  # 黑色
                                2,  # 红色
                                0,  # 绿色
                                1])  # 蓝色
        outpucolor_array = np.array([[0, 0, 0],  # 黑色 3
                                     [0, 0, 255],  # 红色 1
                                     [0, 255, 0],  # 绿色 0
                                     [255, 0, 0]])  # 蓝色 2
        convert_img = convert_1channel_2png_wrapper(jpg_img,color_array,outpucolor_array)
        output_file = outputdir+file_name+'.png'
        cv2.imwrite(output_file,convert_img)

def check_corr_correct(input_dir1,input_dir2):
    import numpy as np
    import os
    b = FileFilt()
    b.FindFile(dirr=input_dir1)
    count = 0
    for k in b.fileList:
        if k == '':
            continue
        print k.split("/")[-1]
        filename_split = k.split("/")[-1].split(".")
        #print filename_split
        if len(filename_split) > 1:
            #       print str(filename_split[-2])
            file_name = str(filename_split[-2])
        else:
            file_name =  str(filename_split[0])
        file_name = file_name.replace('select_ori_original_','')
        check_file_path = input_dir2 +'/'+ '_groundtruth_(1)_select_ori_'+file_name+'.png'
        if os.path.exists(check_file_path):
            print count,filename_split,check_file_path,'exst'
            count+=1
    print 'end'

@jit(nopython=True, cache=True)
def cacalute_image_mean_value_wrapper(jpg_img):
    mean_value = np.zeros(3)
    count = 0
    for i in range(0, jpg_img.shape[0]):
        for j in range(0, jpg_img.shape[1]):
            mean_value[0]+=jpg_img[i, j,0]
            mean_value[1] += jpg_img[i, j, 1]
            mean_value[2] += jpg_img[i, j, 2]
            count+=1
    return mean_value/count

def cacalute_image_mean_value(inpudir):
    import numpy as np

    b = FileFilt()
    b.FindFile(dirr=inpudir)
    mean_value = np.zeros(3)
    count =0
    for k in b.fileList:
        if k == '':
            continue
        print k.split("/")[-1]
        filename_split = k.split("/")[-1].split(".")
        print count,filename_split
        if len(filename_split) > 1:
            #       print str(filename_split[-2])
            file_name = str(filename_split[-2])
        else:
            file_name = str(filename_split[0])
        jpg_img = cv2.imread(k, cv2.IMREAD_COLOR)
        result = cacalute_image_mean_value_wrapper(jpg_img)
        mean_value+=result
        count+=1
    print mean_value,count,mean_value/count

#select_ori_original_0003036e-83a4-471a-9759-09491a05bf9d.png
#_groundtruth_(1)_select_ori_0003036e-83a4-471a-9759-09491a05bf9d.png
def extract_contain_name(inpudir,extract_input_dir,extract_dir):
    import shutil
    key_map = {}
    b = FileFilt()
    b.FindFile(dirr=inpudir)
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
        key_map[file_name] = True

    b.fileList = [""]
    b.FindFile(dirr=extract_input_dir)
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

        if file_name in key_map:
            output_file = extract_dir + k.split("/")[-1]
            shutil.copyfile(k, output_file)

def batch_matting(input_img_path,input_trimap_path,output_dir):
    import scipy.misc
    b = FileFilt()
    b.FindFile(dirr=input_img_path)
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

        result = matting(k,input_trimap_path+file_name+'.png')
        scipy.misc.imsave(output_dir+file_name+'.png', result)


def test_Augmentor():
    import Augmentor
    #p = Augmentor.Pipeline("G:/yuanqing/lfw/lfwdataset/test/extract")
    p = Augmentor.Pipeline("G:/yuanqing/lfw/lfwdataset/png")
    # Point to a directory containing ground truth data.
    # Images with the same file names will be added as ground truth data
    # and augmented in parallel to the original data.
    p.ground_truth("G:/yuanqing/lfw/lfwdataset/png")
    p.set_save_format('png')
    # Add operations to the pipeline as normal:
    p.rotate(probability=0.5, max_left_rotation=20, max_right_rotation=20)
    # p.rotate90(probability=0.5)
    # p.rotate270(probability=0.5)
    # p.flip_left_right(probability=0.5)
    p.rotate_random_90(probability=0.5)
    p.zoom_random(probability=0.5, percentage_area=0.9)
    p.random_distortion(probability=0.2, grid_width=4, grid_height=4, magnitude=8)
    # p.flip_top_bottom(probability=0.5)
    p.sample(10)
def data_Augmentor(source_path ,gt_path,output_directory):
    import Augmentor
    #p = Augmentor.Pipeline("G:/yuanqing/lfw/lfwdataset/test/extract")
    p = Augmentor.Pipeline(source_path,output_directory=output_directory)
    # Point to a directory containing ground truth data.
    # Images with the same file names will be added as ground truth data
    # and augmented in parallel to the original data.
    p.ground_truth(gt_path)
    p.set_save_format('png')
    # Add operations to the pipeline as normal:
    p.rotate(probability=0.5, max_left_rotation=20, max_right_rotation=20)
    # p.rotate90(probability=0.5)
    # p.rotate270(probability=0.5)
    # p.flip_left_right(probability=0.5)
    p.rotate_random_90(probability=0.5)
    p.zoom_random(probability=0.5, percentage_area=0.9)
    p.random_distortion(probability=0.2, grid_width=4, grid_height=4, magnitude=8)
    # p.flip_top_bottom(probability=0.5)
    p.sample(20000)

@jit(nopython=True, cache=True)
def convert_seg_map_4_to3_wraaper(jpg_img):
    convert_img = np.zeros((jpg_img.shape[0], jpg_img.shape[1], 3), dtype=np.uint8)

    for i in range(0, jpg_img.shape[0]):
        for j in range(0, jpg_img.shape[1]):
            color = jpg_img[i, j]
            if color[0] == 0 and color[1] == 0 and color[2] == 255:
                convert_img[i, j] = [0, 0, 255]
            elif color[0] == 0 and color[1] == 255 and color[2] == 0:
                convert_img[i, j] = [0, 255, 0]
            else:
                convert_img[i, j] = [255, 0, 0]
    return convert_img

def convert_seg_map_4_to3(inputdir,outputdir):

    b = FileFilt()
    b.FindFile(dirr=inputdir)
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

        jpg_img = cv2.imread(k, cv2.IMREAD_COLOR)
        convert_img = convert_seg_map_4_to3_wraaper(jpg_img)
        output_file = outputdir+file_name+'.png'
        cv2.imwrite(output_file,convert_img)

def scale_to_250(inputdir,outputdir,use_neaset_scale = False):
    from  fitting.util import  rescale_imge_with_bbox2
    b = FileFilt()
    b.FindFile(dirr=inputdir)
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

        jpg_img = cv2.imread(k, cv2.IMREAD_COLOR)
        #convert_img = rescale_imge_with_bbox2(jpg_img,np.array([-89,0,478,567]),np.array([255,0,0]))
        convert_img = jpg_img
        if use_neaset_scale:
            convert_img = cv2.resize(convert_img, (250, 250), interpolation=cv2.INTER_NEAREST)
        else:
            convert_img =cv2.resize(convert_img,(250,250),interpolation=cv2.INTER_CUBIC)
        output_file = outputdir+file_name+'.png'
        cv2.imwrite(output_file,convert_img)
def rename_aug_source(inputdir):
    import os
    for file in os.listdir(inputdir):
        if os.path.isfile(os.path.join(inputdir, file)) == True:
            newname = file.replace("original_", "")
            os.rename(os.path.join(inputdir, file), os.path.join(inputdir, newname))
def rename_aug_gt(inputdir):
    import os
    for file in os.listdir(inputdir):
        if os.path.isfile(os.path.join(inputdir, file)) == True:
            newname = file.replace("_groundtruth_(1)_", "")
            os.rename(os.path.join(inputdir, file), os.path.join(inputdir, newname))


if __name__ == '__main__':
    #test_Augmentor()
    if 0:
        inpudir = 'G:/yuanqing/lfw/parts_lfw_funneled_gt_images/parts_lfw_funneled_gt_images/'
        outputdir = inpudir+'/png/'
        inpudir = 'G:/yuanqing/lfw/lfwdataset/png/'
        outputdir = 'G:/yuanqing/lfw/lfwdataset/gt_jpg/'
        #convert_png_2_jpg(inpudir, outputdir)
        inpudir = 'G:/yuanqing/lfw/lfwdataset/gt_jpg/'
        outputdir = 'G:/yuanqing/lfw/lfwdataset/gt_convert_png/'
        #convert_jpg_2_png(inpudir, outputdir)
        #convert_ppm_2_png(inpudir, outputdir)    # lfw_input = 'G:/yuanqing/lfw/facehair_seg_lfw-funneled/lfw_funneled/jpg/'
        # lfw_extract = 'G:/yuanqing/lfw/facehair_seg_lfw-funneled/lfw_funneled/jpg/extract/'
        # extract_contain_name(outputdir,lfw_input,lfw_extract)
        # batch_matting('G:/yuanqing/faceproject/zhengjianzhao_seg_dataset/bgBlue/',
        #               'G:/yuanqing/faceproject/zhengjianzhao_seg_dataset/bgBlue_trimap/',
        #               'G:/yuanqing/faceproject/zhengjianzhao_seg_dataset/bgBlue_alpha/'
        #               )
        inpudir = 'G:/yuanqing/lfw/lfwdataset/extract/'
        outputdir = 'G:/yuanqing/lfw/lfwdataset/extract_png/'
        #convert_jpg_2_png(inpudir,outputdir)
        inpudir = 'G:/yuanqing/lfw/lfwdataset/select_png/'
        extract_inpudir = 'G:/yuanqing/lfw/lfwdataset/extract_png/'
        outputdir = 'G:/yuanqing/lfw/lfwdataset/select_ori/'
        inpudir = 'G:/yuanqing/lfw/lfwdataset/select_ori/'
        extract_inpudir = 'G:/yuanqing/lfw/lfwdataset/select_png/'
        outputdir = 'G:/yuanqing/lfw/lfwdataset/select_gt/'
        #extract_contain_name(inpudir,extract_inpudir,outputdir)
        #data_Augmentor(inpudir,outputdir)

        inpudir = 'G:/yuanqing/lfw/lfwdataset/new_generate_gt/'
        outputdir = 'G:/yuanqing/lfw/lfwdataset/new_generate_gt/new_generate_gt_refine/'
        #convert_jpg_2_png_with_color(inpudir,outputdir)
        inpudir = 'G:/yuanqing/lfw/lfwdataset/hair_face_seg_dataset/new_generate_gt_refine/'
        outputdir = 'G:/yuanqing/lfw/lfwdataset/hair_face_seg_dataset/new_generate_gt_refine_1channel/'
        #convert_png2_1channel(inpudir,outputdir)
        inpudir = 'G:/yuanqing/lfw/lfwdataset/hair_face_seg_dataset/select_gt/'
        outputdir = 'G:/yuanqing/lfw/lfwdataset/hair_face_seg_dataset/select_gt_1channel/'
        #convert_png2_1channel(inpudir,outputdir)
        #extract_contain_name(outputdir, lfw_input, lfw_extract)

        inpudir = 'G:/yuanqing/lfw/lfwdataset/hair_face_seg_dataset/new_generate_gt_refine_1channel/'
        outputdir = 'G:/yuanqing/lfw/lfwdataset/hair_face_seg_dataset/new_generate_gt_refine_1channel_validate/'
        #convert_1channel_png2(inpudir,outputdir)

        inpudir = 'G:/yuanqing/lfw/lfwdataset/hair_face_seg_dataset/select_gt_1channel/'
        outputdir = 'G:/yuanqing/lfw/lfwdataset/hair_face_seg_dataset/select_gt_1channel_validate/'
        # convert_1channel_png2(inpudir,outputdir)
        # check_corr_correct('G:/yuanqing/lfw/lfwdataset/hair_face_seg_dataset/new_generate_ori/',
        #                    'G:/yuanqing/lfw/lfwdataset/hair_face_seg_dataset/new_generate_gt_refine/')

        #cacalute_image_mean_value('G:/yuanqing/lfw/lfwdataset/hair_face_seg_dataset/new_generate_ori/')
        inpudir = 'G:/yuanqing/lfw/lfwdataset/hair_face_seg_dataset/bgblue_dataset/bgblue_source/'
        outputdir = 'G:/yuanqing/lfw/lfwdataset/hair_face_seg_dataset/bgblue_dataset/bgblue_source_png/'
        #convert_jpg_2_png(inpudir, outputdir)

        inpudir = 'G:/yuanqing/lfw/lfwdataset/hair_face_seg_dataset/bgblue_dataset/bgblue_source_png/'
        outputdir = 'G:/yuanqing/lfw/lfwdataset/hair_face_seg_dataset/bgblue_dataset/bgblue_source_png_sclae250/'
        #convert_seg_map_4_to3(inpudir,outputdir)
        inpudir = 'G:/yuanqing/lfw/lfwdataset/hair_face_seg_dataset/bgblue_dataset/bgblue_gt_png/'
        outputdir = 'G:/yuanqing/lfw/lfwdataset/hair_face_seg_dataset/bgblue_dataset/bgblue_gt_png_scale250/'
        #scale_to_250(inpudir,outputdir,use_neaset_scale=True)
        source_path= 'G:/yuanqing/lfw/lfwdataset/hair_face_seg_dataset/bgblue_dataset/bgblue_source_png_sclae250/'
        gt_path = 'G:/yuanqing/lfw/lfwdataset/hair_face_seg_dataset/bgblue_dataset/bgblue_gt_png_scale250/'
        output_directory = 'G:/yuanqing/lfw/lfwdataset/hair_face_seg_dataset/bgblue_dataset/aug_output/'
        #data_Augmentor(source_path ,gt_path,output_directory)
        #rename_aug_source('G:/yuanqing/lfw/lfwdataset/hair_face_seg_dataset/bgblue_dataset/aug_source/')
        #rename_aug_gt('G:/yuanqing/lfw/lfwdataset/hair_face_seg_dataset/bgblue_dataset/aug_gt/')

        inpudir = 'G:/yuanqing/lfw/lfwdataset/hair_face_seg_dataset/bgblue_dataset/aug_gt/'
        outputdir = 'G:/yuanqing/lfw/lfwdataset/hair_face_seg_dataset/bgblue_dataset/aug_gt_refine/'
        #convert_jpg_2_png_with_color(inpudir,outputdir)
        inpudir = 'G:/yuanqing/lfw/lfwdataset/hair_face_seg_dataset/bgblue_dataset/aug_gt_refine/'
        outputdir = 'G:/yuanqing/lfw/lfwdataset/hair_face_seg_dataset/bgblue_dataset/aug_gt_refine_one_channel/'
        #convert_png2_1channel(inpudir, outputdir)
        inpudir = 'G:/yuanqing/lfw/lfwdataset/hair_face_seg_dataset/bgblue_dataset/bgblue_gt_png_scale250/'
        outputdir = 'G:/yuanqing/lfw/lfwdataset/hair_face_seg_dataset/bgblue_dataset/bgblue_gt_png_scale250_one_channel/'
        #convert_png2_1channel(inpudir, outputdir)
        cacalute_image_mean_value('G:/yuanqing/lfw/lfwdataset/hair_face_seg_dataset/bgblue_dataset/aug_source/')

    inpudir = 'C:\Users\hehua2015\Pictures/testpaper/'
    outputdir = 'C:\Users\hehua2015\Pictures/testpaper/test_convert/'
    #convert_jpg_2_png(inpudir, outputdir)
    inpudir = 'C:\Users\hehua2015\Pictures/testpaper/test_convert\crop/'
    outputdir = 'C:\Users\hehua2015\Pictures/testpaper/test_convert\crop\crop250/'
    scale_to_250(inpudir, outputdir, use_neaset_scale=False)