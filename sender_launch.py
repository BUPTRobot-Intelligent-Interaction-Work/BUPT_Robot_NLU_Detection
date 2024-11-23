import requests
import json
import base64
import cv2
import pprint
#PyTorch_REST_API_URL = 'http://10.11.33.205:4003/xuni/inference'
PyTorch_REST_API_URL = 'http://algm-221974471452315648.competition.harix.iamidata.com/xuni/inference'

def launch_model(apath):
    thisid = '1'
    payload = {'action':'initialize net', 'thisid':thisid}
    payload = json.dumps(payload)
    r = requests.post(PyTorch_REST_API_URL, data=payload).json()
    print(r)

launch_model(apath='a model path')

