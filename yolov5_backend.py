import cv2
import numpy as np
import pygetwindow as gw
import pyautogui
from PIL import Image
from models.experimental import attempt_download

import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

import threading
import time

import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
from utils.datasets import letterbox



device = select_device('0')
half = device.type != 'cpu'  # half precision only supported on CUDA

# Load model
model_human_detection = attempt_load('person_best.pt', map_location=device)  # load FP32 model
model_action_detection = attempt_load('best_action.pt', map_location=device)  # load FP32 model
model_exhibit_detection = attempt_load('best_exhibit.pt', map_location=device)  # load FP32 model
model_face_detection = attempt_load('best_face.pt', map_location=device)  # load FP32 model
model_gesture_detection = attempt_load('best_gesture.pt', map_location=device)  # load FP32 model
model_emotion_detection = attempt_load('best_emotion.pt', map_location=device)  # load FP32 model


def detect(frame, imgsz, augment, conf_thres, iou_thres, classes, agnostic_nms, save_txt, save_img, view_img, webcam, opt, im0s, save_dir, classify, modelc, model):
    img0 = frame
    img = letterbox(img0, new_shape=imgsz)[0]
    img = img.transpose(2, 0 ,1)
    # global model

    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    # print(imgsz)
    if half:
        model.half()  # to FP16


    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    # if device.type != 'cpu':
    #     model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()

    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # print(img.shape)
    # Inference
    t1 = time_synchronized()
    pred = model(img, augment=augment)[0]
    # print(pred)
    # Apply NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
    t2 = time_synchronized()

    # print(pred)
    # Process detections
    for i, det in enumerate(pred):  # detections per image

        s, im0, frame = '', img0, frame
        s += '%gx%g ' % img.shape[2:]  # print string
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

            # Write results
            for *xyxy, conf, cls in reversed(det):
                if save_txt:  # Write to file
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format

                if save_img or view_img:  # Add bbox to image
                    label = f'{names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

        # Print time (inference + NMS)
        # print(f'{s}Done. ({t2 - t1:.3f}s)')
        # print()
        # Stream results
        if view_img:
            cv2.imshow("", im0)
            cv2.waitKey(1)  # 1 millisecond

    print(frame.shape)
    S_last = 0.0
    xywh = 0
    label = 0
    # print(pred)
    pred = pred[0]
    for pre in pred:
        print(pre)
        S_ = (pre[2]-pre[0]).item() * (pre[3]-pre[1]).item()
        if S_ > S_last:
            S_last = S_
            xyxy = pre[:4].tolist()
            xywh = [xyxy[0], xyxy[1], xyxy[2]-xyxy[0], xyxy[3]-xyxy[1]]
            # xywh_yolo = [xywh[0], xywh[1], xywh[2], xywh[3]]
            label = pre[5].item()
    num = pred.shape[0]

    if xywh == 0:
        return (False, xywh, label, num)
    return (True, xywh, label, num)


        

        
        # return 




app = Flask(__name__)
CORS(app, supports_credentials=True)

frame = None

def capture_window(window_title):
    global frame
    try:
        # 获取特定窗口
        win = gw.getWindowsWithTitle(window_title)[0]
        win.activate()  # 将窗口带到前台（可选）
        # 循环捕捉和显示窗口内容
        while True:
            # 使用 pyautogui 截取窗口的图像
            img = pyautogui.screenshot(region=(15, 360, 1500, 1060))
            # 将图片转换为 numpy 数组，以便 OpenCV 可以处理
            frame_temp = np.array(img)
            frame = cv2.cvtColor(frame_temp, cv2.COLOR_BGR2RGB)  # 转换颜色从RGB到BGR
            # 显示图像
            cv2.imshow(window_title, frame)

            # 按 'q' 键退出
            if cv2.waitKey(1) == ord('q'):
                break

        # 关闭所有 OpenCV 窗口
        cv2.destroyAllWindows()
    except Exception as e:
        print("Error:", e)



num2label_face = {
    "0": "face",
    "1": "kezundi",
    "2": "wuyuhao",
    "3": "xiaguoyang"
}

num2label_exhibit = {
    "0": "monalisa",
    "1": "star_night",
    "2": "sunflower",
    "3": "scream"
}

num2label_action = {
    "0": "hugging",
    "1": "sitting",
    "2": "using_laptop",
    "3": "clapping",
    "4": "calling",
    "5": "drinking"
}

num2label_gesture = {
    '0': 'like', 
    '1': 'fist', 
    '2':'ok', 
    '3':'one', 
    '4':'two', 
    '5':'three', 
    '6':'four', 
    '7':'palm', 
    '8': 'no_gesture'
}

num2label_emotion = {'0':'happy', '1':'neutral', '2':'sad'}

@app.route('/human_detection', methods=['GET', 'POST'])
def human_detection():
    global frame
    res = detect(frame, imgsz=640, 
        augment=False, conf_thres=0.25, iou_thres=0.45, classes=None, 
        agnostic_nms=False, save_txt=False, save_img=False, 
        view_img=True, webcam=False, opt=None, im0s=None, save_dir=None, classify=False, modelc=None, model=model_human_detection)

    return jsonify({"detection": res[0], "xywh": res[1], "label": res[2], "num": res[3]})
    

@app.route('/face_detection', methods=['GET', 'POST'])
def face_detection():
    global frame
    res = detect(frame, imgsz=640, 
        augment=False, conf_thres=0.25, iou_thres=0.45, classes=None, 
        agnostic_nms=False, save_txt=False, save_img=False, 
        view_img=True, webcam=False, opt=None, im0s=None, save_dir=None, classify=False, modelc=None, model=model_face_detection)
    
    return jsonify({"detection": res[0], "xywh": res[1], "label": num2label_face[str(int(res[2]))], "num": res[3]})

@app.route('/action_detection', methods=['GET', 'POST'])
def action_detection():
    global frame
    res = detect(frame, imgsz=640, 
        augment=False, conf_thres=0.25, iou_thres=0.45, classes=None, 
        agnostic_nms=False, save_txt=False, save_img=False, 
        view_img=True, webcam=False, opt=None, im0s=None, save_dir=None, classify=False, modelc=None, model=model_action_detection)
    
    return jsonify({"detection": res[0], "xywh": res[1], "label": num2label_action[str(int(res[2]))], "num": res[3]})

@app.route('/exhibit_detection', methods=['GET', 'POST'])
def exhibit_detection():
    global frame
    res = detect(frame, imgsz=640, 
        augment=False, conf_thres=0.25, iou_thres=0.45, classes=None, 
        agnostic_nms=False, save_txt=False, save_img=False, 
        view_img=True, webcam=False, opt=None, im0s=None, save_dir=None, classify=False, modelc=None, model=model_exhibit_detection)
    print(res)
    return jsonify({"detection": res[0], "xywh": res[1], "label": num2label_exhibit[str(int(res[2]))], "num": res[3]})

@app.route('/gesture_detection', methods=['GET', 'POST'])
def gesture_detection():
    global frame
    res = detect(frame, imgsz=640, 
        augment=False, conf_thres=0.25, iou_thres=0.45, classes=None, 
        agnostic_nms=False, save_txt=False, save_img=False, 
        view_img=True, webcam=False, opt=None, im0s=None, save_dir=None, classify=False, modelc=None, model=model_gesture_detection)
    
    return jsonify({"detection": res[0], "xywh": res[1], "label": num2label_gesture[str(int(res[2]))], "num": res[3]})

@app.route('/emotion_detection', methods=['GET', 'POST'])
def emotion_detection():
    global frame
    res1 = detect(frame, imgsz=640, 
        augment=False, conf_thres=0.25, iou_thres=0.45, classes=None, 
        agnostic_nms=False, save_txt=False, save_img=False, 
        view_img=True, webcam=False, opt=None, im0s=None, save_dir=None, classify=False, modelc=None, model=model_face_detection)
    if not res1[0]:
        return jsonify({"detection": False, "xywh": 0, "label": "no_face", "num": 0})
    print(res1[1])
    # frame_2 = frame[int(res1[1][0]):int(res1[1][0]+res1[1][2]), int(res1[1][1]):int(res1[1][1]+res1[1][3])]
    frame_2 = frame[int(res1[1][1]):int(res1[1][1]+res1[1][3]), int(res1[1][0]):int(res1[1][0]+res1[1][2])]
    print(type(frame_2))
    print(frame_2.shape)
    # 转成灰度图
    frame_2 = cv2.cvtColor(frame_2, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("test.jpg", frame_2)
    frame_2 = cv2.imread("test.jpg", flags=cv2.IMREAD_COLOR)

    res2 = detect(frame_2, imgsz=640, 
        augment=False, conf_thres=0.25, iou_thres=0.45, classes=None, 
        agnostic_nms=False, save_txt=False, save_img=False, 
        view_img=True, webcam=False, opt=None, im0s=None, save_dir=None, classify=False, modelc=None, model=model_emotion_detection)
    print(res2)

    return jsonify({"detection": res2[0], "xywh": res2[1], "label": num2label_emotion[str(int(res2[2]))], "num": res2[3]})

def task_read_video_stream():
    capture_window("SDK demo")

def task_backend():
    app.run(host="0.0.0.0", port=8002)

if __name__ == '__main__':
    # 创建线程实例
    thread1 = threading.Thread(target=task_read_video_stream)
    thread2 = threading.Thread(target=task_backend)

    # 启动线程
    thread1.start()
    thread2.start()

    thread1.join()
    thread2.join()
