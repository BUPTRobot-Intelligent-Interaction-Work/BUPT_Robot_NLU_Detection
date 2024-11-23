import requests
import json
import base64
import cv2
import pprint
#PyTorch_REST_API_URL = 'http://10.11.33.205:4003/xuni/inference'
PyTorch_REST_API_URL = 'http://algm-221974471452315648.competition.harix.iamidata.com/xuni/inference'

def cv2_base64(image):
    base64_str = cv2.imencode('.jpg',image)[1].tostring()
    base64_str = base64.b64encode(base64_str).decode('utf8')
    return base64_str
    
def predict_result(image_path):
    image = cv2.imread(image_path)
    #image = cv2.imencode('.jpg', image)
    #payload = {'action':'inference', 'image': image[1].tolist()}
    payload = {'action':'inference', 'image': cv2_base64(image)}
    payload = json.dumps(payload)
    r = requests.post(PyTorch_REST_API_URL, data=payload).json()
    print(r)


for i in range(1):
    predict_result('data/images/RGB17.jpg')


