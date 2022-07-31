#!/usr/bin/env python
# encoding: utf-8
#
"""
@software: PyCharm
@time: 2022/7/31 15:05
@desc:
"""
from fitting.util import FileFilt
from face_generate import generate_face
from strand_convert import build_hair_for_img_simgle


def generate_shape(project_dir="", gender_map={}):
    """
    generate shape for natural expression
    :return:
    """
    # project_dir = 'E:\workspace/vrn_data\hairstyle/man/'
    # project_dir = 'E:\workspace/vrn_data\paper_select1/man/'
    # project_dir = 'E:\workspace/vrn_data\paper_compare/neatrual_1/man/'
    # project_dir = 'D:\huayunhe/facewarehouse_new/FaceWarehouse_neutral_img_b/male/'

    b = FileFilt()
    b.FindFile(dirr=project_dir)

    for k in b.fileList:
        if k == '':
            continue
            #       print k.split("/")[-1]
        # print k.split("/")
        filename_split = k.split("/")[-1].split(".")
        #        print filename_split
        if len(filename_split) > 1:
            print str(filename_split[-2])
            file_name = str(filename_split[-2])
            fomat_name = str(filename_split[-1])
        else:
            file_name = str(filename_split[0])
            fomat_name = ''
        obj_name = file_name
        # if os.path.exists(project_dir+obj_name+'/generate_face/'+'face_with_texture.obj'):
        #     continue
        gender = gender_map.get(obj_name, 'female')
        pre_result = generate_face(frame_model_path='./models/%s_model.pkl' % gender,
                                   vrn_object_dir=project_dir + obj_name + '/',
                                   object_name=obj_name,
                                   project_dir=project_dir,
                                   out_put_dir=project_dir + obj_name + '/generate_face/', use_3d_landmark=False)


def build_hair(img_dir, landmark_dir=None, skipfilename_list=[]):
    """
    :return:
    """
    if landmark_dir is None:
        landmark_dir = img_dir + "/Landmark"
    import cv2
    b = FileFilt()
    b.FindFile(dirr=img_dir)
    for k in b.fileList:
        if k == '':
            continue
            #       print k.split("/")[-1]
        # print k.split("/")
        filename_split = k.split("/")[-1].split(".")
        #        print filename_split
        if len(filename_split) > 1:
            print str(filename_split[-2])
            file_name = str(filename_split[-2])
            fomat_name = str(filename_split[-1])
        else:
            file_name = str(filename_split[0])
            fomat_name = ''
        # if 'A13010436665609' != file_name:
        #      continue
        if 'directed_color' == file_name or 'undirected_color' == file_name:
            continue
        # if file_name in skip_file:
        #     continue
        if file_name in skipfilename_list:
            pass
        else:
            continue
        build_hair_for_img_simgle(object_name=file_name,
                                  input_ori_img_file=img_dir + '/' + file_name + '.' + fomat_name,
                                  input_seg_img_file=img_dir + 'Seg_refined/' + file_name + '.png',
                                  input_dir_img_file=img_dir + 'Strand/' + file_name + '.png',
                                  #           input_landmark_file = landmark_dir+file_name+'/'+'2d/'+file_name+'.txt',
                                  input_landmark_file=landmark_dir + file_name + '.txt',
                                  out_put_dir=img_dir + 'result/' + file_name + '/' + 'builded_hair/',
                                  project_dir=img_dir)


def testcase1():
    gender_map = {"06010": "male", "06018": "male", "06021": "female", "06076": "male", "06132": "female",
                  "06206": "female"}
    generate_shape("E:/workspace/vrn_data/avata-testdata/", gender_map=gender_map)


def testcase2():
    build_hair("img_dir")


def run():
    testcase1()


if __name__ == '__main__':
    run()
