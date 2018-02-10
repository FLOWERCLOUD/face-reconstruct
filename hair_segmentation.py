# -- coding: utf-8 --
import numpy as np
import cv2
from fitting.util import FileFilt
from  bayesian_matting_master.bayesian_matting import matting

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



if __name__ == '__main__':
    inpudir = 'G:/yuanqing/lfw/parts_lfw_funneled_gt_images/parts_lfw_funneled_gt_images/'
    outputdir = inpudir+'/png/'
    #convert_ppm_2_png(inpudir, outputdir)    # lfw_input = 'G:/yuanqing/lfw/facehair_seg_lfw-funneled/lfw_funneled/jpg/'
    # lfw_extract = 'G:/yuanqing/lfw/facehair_seg_lfw-funneled/lfw_funneled/jpg/extract/'
    # extract_contain_name(outputdir,lfw_input,lfw_extract)
    batch_matting('G:/yuanqing/faceproject/zhengjianzhao_seg_dataset/bgBlue/',
                  'G:/yuanqing/faceproject/zhengjianzhao_seg_dataset/bgBlue_trimap/',
                  'G:/yuanqing/faceproject/zhengjianzhao_seg_dataset/bgBlue_alpha/'
                  )



