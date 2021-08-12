# coding:utf-8

import sys 
sys.path.append('/home/aistudio/external-libraries')

import argparse
import dlib
import numpy as np
from copy import deepcopy
import cv2
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import torch
import torch.nn.functional as F
from process.data_helper import RESIZE_SIZE
from process.augmentation import color_augumentor

#http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat
#http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

class FaceRecognitionExample(object):
    def __init__(self, img_1, img_2, img_3):
        super(FaceRecognitionExample, self).__init__()
        self.img_1 = img_1
        self.img_2 = img_2
        self.img_3 = img_3
        self.detector = dlib.get_frontal_face_detector()
        self.img_size = 150
        self.predictor = dlib.shape_predictor(r'./alive_detect/shape_predictor_68_face_landmarks.dat')
        self.recognition = dlib.face_recognition_model_v1('alive_detect/dlib_face_recognition_resnet_model_v1.dat')

    def point_draw(self, img, sp, title, save):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        for i in range(68):
            cv2.putText(img, str(i), (sp.part(i).x, sp.part(i).y), cv2.FONT_HERSHEY_DUPLEX, 0.3, (0, 0, 255), 1,
                        cv2.LINE_AA)
            # cv2.drawKeypoints(img, (sp.part(i).x, sp.part(i).y),img, [0, 0, 255])
        if save:
            #filename = title+str(np.random.randint(100))+'.jpg'
            filename = title+'.jpg'
            cv2.imwrite(filename, img)
        os.system("open %s"%(filename)) 

    def show_origin(self, img):
        cv2.imshow('origin', img)
        cv2.waitKey(0)
        cv2.destroyWindow('origin')

    def getfacefeature(self, img):
        image = dlib.load_rgb_image(img)
        ## 人脸对齐、切图
        # 人脸检测
        dets = self.detector(image, 1)
        if len(dets) == 1:
            # 关键点提取
            shape = self.predictor(image, dets[0])
            print("Computing descriptor on aligned image ..")
            #人脸对齐 face alignment
            images = dlib.get_face_chip(image, shape, size=self.img_size)

            self.point_draw(image, shape, 'before_image_warping', save=True)
            shapeimage = np.array(images).astype(np.uint8)
            dets = self.detector(shapeimage, 1)
            if len(dets) == 1:
                point68 = self.predictor(shapeimage, dets[0])
                self.point_draw(shapeimage, point68, 'after_image_warping', save=True)

            #计算对齐后人脸的128维特征向量
            face_descriptor_from_prealigned_image = self.recognition.compute_face_descriptor(images)
        return face_descriptor_from_prealigned_image

    def compare(self):
        vec1 = np.array(self.getfacefeature(self.img_1))
        vec2 = np.array(self.getfacefeature(self.img_2))
        vec3 = np.array(self.getfacefeature(self.img_3))

        same_people = np.sqrt(np.sum((vec2-vec3)*(vec2-vec3)))
        not_same_people12 = np.sqrt(np.sum((vec1-vec2)*(vec1-vec2)))
        not_same_people13 = np.sqrt(np.sum((vec1-vec3)*(vec1-vec3)))
        print('distance between different people12:{:.3f}, different people13:{:.3f}, same people:{:.3f}'.\
              format(not_same_people12, not_same_people13, same_people))


## 活体检测模型
class FaceSpoofing(object):

    def __init__(self, f):
        from model.model_baseline import Net

        self.img_size = 48
        self.device = torch.device('cuda' if torch.has_cuda else 'cpu')
        self.net = Net(num_class=2, is_first_bn=True).to(self.device)

        state_dict = torch.load(f, map_location=self.device)
        self.net.load_state_dict({
            k[7:] if k.startswith('module.') else k: v for k,v in state_dict.items()
        })

  # 实现活体检测二分类
    def classify(self, face_align):
        color = cv2.resize(face_align, (RESIZE_SIZE, RESIZE_SIZE))
        color = color_augumentor(color, target_shape=(self.img_size, self.img_size, 3), is_infer=True)
        num_patch = len(color)
        color = np.concatenate(color, axis=0)

        image = np.transpose(color, (0, 3, 1, 2))
        image = image.astype(np.float32)
        image = image / 255.0

        input_image = torch.FloatTensor(image).to(self.device)

        with torch.no_grad():
            logit, *_ = self.net(input_image)
            logit = torch.mean(logit, dim = 0, keepdim = False)

            return np.argmax(logit.detach().cpu().numpy())


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--alive_detect_model', required=True)
    parser.add_argument('--key_point_detect_model', required=True)
    parser.add_argument('--detect_vidio', required=True)
    parser.add_argument('--detect_res', required=True)
    args = parser.parse_args()

    # 初始化人脸检测模型
    detector = dlib.get_frontal_face_detector()
    # 初始化活体检测模型
    face_spoofing = FaceSpoofing(args.alive_detect_model)
    # 初始化关键点检测模型
    predictor = dlib.shape_predictor(args.key_point_detect_model)
    # 从摄像头读取图像, 若摄像头工作不正常，可使用：cv2.VideoCapture("demo.mp4"),从视频中读取图像
    cap = cv2.VideoCapture(args.detect_vidio) # cap = cv2.VideoCapture(0)
    _, idemo = cap.read()
    fourcc = cv2.VideoWriter_fourcc('I', '4', '2', '0')		#保存为avi格式视频
    out = cv2.VideoWriter(args.detect_res, fourcc, 30.0, (idemo.shape[1],idemo.shape[0]), True)	#30.0为保存视频帧数

    while cap.isOpened():
        # 读取图片
        ret, frame_src = cap.read()
        if not ret:
            break
        # 将图片缩小为原来大小的1/3
        cam_h, cam_w = frame_src.shape[0:2]
        frame = cv2.resize(frame_src, (int(cam_w / 3), int(cam_h / 3)))
        #face_align = frame
        # 使用检测模型对图片进行人脸检测
        dets = detector(frame, 1)
        # 遍历检测结果
        for det in dets:
            # 对检测到的人脸提取人脸关键点
            shape=predictor(frame, det)
            # 人脸对齐
            face_align=dlib.get_face_chip(frame, shape, 150,0.1)
            ## 活体检测
            if not face_spoofing.classify(face_align):
                # 框为红色
                frame=cv2.rectangle(frame,(det.left(),det.top()),(det.right(),det.bottom()),(0,0,255),2)
                cv2.putText(frame,"Not Alive",(det.left(),det.top()),cv2.FONT_HERSHEY_DUPLEX, 0.3, (0, 255, 255), 1,cv2.LINE_AA)
            else:
                # 框为绿色
                frame=cv2.rectangle(frame,(det.left(),det.top()),(det.right(),det.bottom()),(0,255,0),2)
                cv2.putText(frame,"Alive",(det.left(),det.top()),cv2.FONT_HERSHEY_DUPLEX, 0.3, (0, 255, 255), 1,cv2.LINE_AA)

        out_frame = cv2.resize(frame,(cam_w,cam_h))
        out.write(out_frame)
    cap.release()
    out.release()

