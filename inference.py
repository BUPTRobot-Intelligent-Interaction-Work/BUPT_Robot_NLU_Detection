import argparse


from utils.datasets import *
from utils.general import *
from utils import *

ONNX_EXPORT = False

parser = argparse.ArgumentParser()
parser.add_argument('--weights', type=str, default='/root/jiaoyu_yolov5/run/weights/best_jiaoyu.pt', help='model.pt path')
parser.add_argument('--source', type=str, default='', help='source')  # file/folder, 0 for webcam
parser.add_argument('--output', type=str, default='', help='output folder')  # output folder
parser.add_argument('--img-size', type=int, default=608, help='inference size (pixels)')
parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
parser.add_argument('--half', type=bool, default=True, help='half precision FP16 inference')
parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
parser.add_argument('--view-img', action='store_true', help='display results')
parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
parser.add_argument('--augment', action='store_true', help='augmented inference')
opt = parser.parse_args()

def net_init(weight_path='/root/jiaoyu_yolov5/run/weights/best_jiaoyu.pt', device='0'):
    imgsz, img_size = opt.img_size, opt.img_size
    weights = weight_path
    device = torch_utils.select_device(device)    
    try:
        assert os.path.exists(weights)
    except:
        print('training did not end properly')
        weights = './best_jiaoyu.pt'
    out, source, half, view_img, save_txt = opt.output, opt.source, opt.half, opt.view_img, opt.save_txt
    half = half and device.type!= 'cpu'  # half precision only supported on CUDA
    model = torch.load(weights, map_location=device)['model']
    model.eval()
    if half:
        model.half()
    else:
        model.float()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img.float()) if device != 'cpu' else None  # run once
    return model

def inference(model, img0, device='cpu'):
    imgsz, img_size = opt.img_size, opt.img_size
    out, source, weights, half, view_img, save_txt = opt.output, opt.source, opt.weights, opt.half, opt.view_img, opt.save_txt
    device = torch_utils.select_device(device)
    #print(device, device.type)
    half = half and device.type!= 'cpu'  # half precision only supported on CUDA
    
    # Set Dataloader

    # Get names and colors
    names = model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
    
    classes = []
    bboxes = []
    confes = []
    # Run inference
    t0 = time.time()
    
    img = letterbox(img0, new_shape=img_size)[0]
    
    #img = img0
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    #img = img[:, :, :].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

        # Inference
    t1 = torch_utils.time_synchronized()
    pred = model(img, augment=opt.augment)[0]
    t2 = torch_utils.time_synchronized()

        # to float
    if half:
        pred = pred.float()

        # Apply NMS
    pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

        # Process detections
    for i, det in enumerate(pred):  # detections per image
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class

            # Write results
            for *xyxy, conf, cls in det:
                classes.append(int(cls))
                bboxes.append([int(xyxy[0]),int(xyxy[1]),int(xyxy[2]),int(xyxy[3])])
                confes.append(float(conf))

            # Print time (inference + NMS)
        print('inference Done. (%.3fs)' % (t2 - t1))
    print('all Done. (%.3fs)' % (time.time() - t0))
    return {'classes': classes, 'bboxes': bboxes, 'confes' : confes}
