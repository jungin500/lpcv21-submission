import os
import platform
from pathlib import Path
# from yolox.data.datasets.coco_classes import COCO_CLASSES
from yolox.predictor import Predictor
from yolox.utils.model_utils import fuse_model
from yolox.exps.base.build import get_exp

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np

# https://github.com/pytorch/pytorch/issues/3678
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path+'/yolov5')

from yolox.data.datasets.lpcvloader import LoadStreams, LoadImages
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort

import time
import yaml
import solution
import main

BALLPERSON_CLASSES = (
    "person",
    "sports ball"
)

def time_synchronized():
    # pytorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()

def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)


def detect(opt, device, half, colorDict, save_img=False):
    out, source, weights, view_img, save_txt, imgsz, skipLimit = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, opt.skip_frames
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    groundtruths_path = opt.groundtruths
    colorOrder = [color for color in colorDict]
    frame_num = 0
    framestr = 'Frame {frame}'
    fpses = []
    frame_catch_pairs = []
    ball_person_pairs = {}
    id_mapping = {}

    for color in colorDict:
        ball_person_pairs[color] = 0
    
    print("FRAMES SKIPPED: " + str(skipLimit))

    # Read Class Name Yaml
    with open(opt.data) as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)
    names = data_dict['names']

    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    deepsort = DeepSort(dir_path + '/' + cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE, 
                        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE, 
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET, use_cuda=True)

    # Initialize
    if not os.path.exists(out):
        os.makedirs(out)  # make new output folder

    # Load model (YOLOv5)
    # model = attempt_load(weights, map_location=device)  # load FP32 model
    # stride = int(model.stride.max())  # model stride
    # imgsz = check_img_size(imgsz, s=stride)  # check img_size

    # Load model (YOLOX)
    EXP_PATH = 'solution/yolox/exps/default/nano.py'
    CHECKPOINT_PATH = 'solution/yolox/weights/yolox_nano.pth'
    
    exp = get_exp(EXP_PATH, None)
    # exp.test_conf = 0.25
    # exp.nmsthre = 0.75
    # exp.test_size = (416, 416)
    model = exp.get_model()
    model.eval()
    
    # Load weight from checkpoint
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(ckpt['model'])

    # Fuse model for optimized inference
    model = fuse_model(model)
    model.to(device)

    # Create predictor from model
    predictor = Predictor(model, exp, BALLPERSON_CLASSES, None, 'gpu' if device.type == 'cuda' else 'cpu', False)

    stride = 32
    imgsz = 640

    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    # names = model.module.names if hasattr(model, 'module') else model.names
    # colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
    # print(names)
    # print(colors)
    '''
        ['person', 'sports ball']
        [[203, 20, 169], [136, 240, 229]]
    '''
    names = ['person', 'sports ball']
    colors = [[203, 20, 169], [136, 240, 229]]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    #Skip Variables
    skipThreshold = 0 #Current number of frames skipped
    
    for path, im0s, vid_cap in dataset:
        if frame_num > 10 and skipThreshold < skipLimit:
            skipThreshold = skipThreshold + 1
            frame_num += 1
            continue
        
        skipThreshold = 0

        # Inference
        t1 = time_synchronized()

        # Inference + NMS (YOLOX)
        pred, img_info = predictor.inference(im0s)

        # YOLOX postprocessing
        cls_conf = predictor.confthre
        ratio = img_info['ratio']

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            s += '384x640 '  # print string
            # gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                # det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                bbox_xywh = []
                confs = []
                clses = []

                bboxes_all = det[:, :4] / ratio
                conf_all = det[:, 4:5] * det[:, 5:6]
                clses_all = det[:, 6:7]

                # Write results
                for bbox_id in range(bboxes_all.shape[0]):
                    x0, y0, x1, y1 = bboxes_all[bbox_id]
                    conf = conf_all[bbox_id]
                    cls = clses_all[bbox_id]

                    if conf < cls_conf:
                        continue

                    img_h, img_w, _ = im0.shape  # get image shape
                    x_c, y_c, bbox_w, bbox_h = main.bbox_rel(img_w, img_h, *[x0, y0, x1, y1])
                    obj = [x_c, y_c, bbox_w, bbox_h]

                    bbox_xywh.append(obj)
                    confs.append([conf.item()])
                    clses.append([cls.item()])
                    
                xywhs = torch.Tensor(bbox_xywh)
                confss = torch.Tensor(confs)
                clses = torch.Tensor(clses)
                # Pass detections to deepsort
                outputs = []
                if not 'disable' in groundtruths_path:
                    # print('\nenabled', groundtruths_path)
                    groundtruths = solution.load_labels(groundtruths_path, img_w,img_h, frame_num)
                    if (groundtruths.shape[0]==0):
                        outputs = deepsort.update(xywhs, confss, clses, im0)
                    else:
                        # print(groundtruths)
                        xywhs = groundtruths[:,2:]
                        tensor = torch.tensor((), dtype=torch.int32)
                        confss = tensor.new_ones((groundtruths.shape[0], 1))
                        clses = groundtruths[:,0:1]
                        outputs = deepsort.update(xywhs, confss, clses, im0)
                    
                    
                    if frame_num >= 2:
                        for real_ID in groundtruths[:,1:].tolist():
                            for DS_ID in xyxy2xywh(outputs[:, :5]):
                                if (abs(DS_ID[0]-real_ID[1])/img_w < 0.005) and (abs(DS_ID[1]-real_ID[2])/img_h < 0.005) and (abs(DS_ID[2]-real_ID[3])/img_w < 0.005) and(abs(DS_ID[3]-real_ID[4])/img_w < 0.005):
                                    id_mapping[DS_ID[4]] = int(real_ID[0])
                else:
                    outputs = deepsort.update(xywhs, confss, clses, im0)


                # draw boxes for visualization
                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, 4]
                    clses = outputs[:, 5]
                    scores = outputs[:, 6]
                    
                    #Temp solution to get correct id's 
                    mapped_id_list = []
                    for ids in identities:
                        if(ids in id_mapping):
                            mapped_id_list.append(int(id_mapping[ids]))
                        else:
                            mapped_id_list.append(ids)

                    ball_detect, frame_catch_pairs, ball_person_pairs = solution.detect_catches(im0, bbox_xyxy, clses, mapped_id_list, frame_num, colorDict, frame_catch_pairs, ball_person_pairs, colorOrder, save_img)
    
                    t3 = time_synchronized()
                    if save_img or view_img:
                        main.draw_boxes(im0, bbox_xyxy, [names[i] if i < len(names) else 'Unknown' for i in clses], scores, ball_detect, id_mapping, identities)
                else:
                    t3 = time_synchronized()


            #Inference Time
            fps = (1/(t3 - t1))
            fpses.append(fps)
            print('FPS=%.2f' % fps)
            
            
            # Stream results
            if view_img:
                cv2.imshow(p, cv2.resize(im0, [1920, 1080]))
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    #Draw frame number
                    tmp = framestr.format(frame = frame_num)
                    t_size = cv2.getTextSize(tmp, cv2.FONT_HERSHEY_PLAIN, 2 , 2)[0]
                    cv2.putText(im0, tmp, (0, (t_size[1] + 10)), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)

                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
                    vid_writer.write(im0)
            frame_num += 1
                    
        
    avgFps = (sum(fpses) / len(fpses))
    print('Average FPS = %.2f' % avgFps)


    outpath = os.path.basename(source)
    outpath = outpath[:-4]
    outpath = out + '/' + outpath + '_out.csv'
    print(outpath)
    solution.write_catches(outpath, frame_catch_pairs, colorDict, colorOrder)

    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)

    return
