import os
import cv2
import time
import argparse
import torch
import warnings
import numpy as np
import time
import base64

from inference import *


env_cpu='0'
net = net_init('weights/yolov5s.pt', env_cpu)
        
def base64_cv2(base64_str):
    imgString = base64.b64decode(base64_str)
    nparr = np.fromstring(imgString,np.uint8)  
    image = cv2.imdecode(nparr,cv2.IMREAD_COLOR)
    return image

def decode_image(image):
    image = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_ANYCOLOR)
    return image

    
import flask
app = flask.Flask(__name__)
@app.route("/xuni/inference", methods=["POST"])
def inference_a_img():
    thisid = 0
    data = flask.request.get_json(force=True)

    root_dir = './'
    log_dir = '{}/train/{}_run'.format(root_dir, thisid)
    try:
        assert net
        astr = 'net initialized'
    except:
        print('training did not end properly')
        return {'code':'-1'}

    time0=time.time()
    base64_str=data['image']
    ori_im=base64_cv2(base64_str)
    time1 = time.time()
    print('start inference')
    results = inference(net, ori_im, env_cpu)
    time2 = time.time()
    print('inference done')
    results['request time'] = time1 - time0
    return results
    
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='deployment relatives')
    parser.add_argument('--aip', type=str, default='0.0.0.0',
                        help='deployment ip')
    parser.add_argument('--aport', type=int, default=3003, help='deployment ip')
    args = parser.parse_args()
    aip = args.aip
    aport = args.aport
    app.run(host=aip, port=aport)
