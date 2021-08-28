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

from yolox.data.datasets.lpcvloader import LoadImages
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

    # Load model (YOLOX)
    EXP_PATH = 'solution/yolox/exps/custom/model_yj.py'
    CHECKPOINT_PATH = 'solution/yolox/weights/best_ckpt.pth'
    BATCH_SIZE=4

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
    # model = fuse_model(model)
    model.to(device)

    model = fuse_model(model)

    # torch.backends.quantized.engine = 'qnnpack'
    # model.qconfig = torch.quantization.get_default_qconfig('qnnpack')
    # model_p = torch.quantization.prepare(model)
    # model = torch.quantization.convert(model_p)
    
    # Create predictor from model
    predictor = Predictor(model, exp, BALLPERSON_CLASSES, device.type, False)

    stride = 32
    imgsz = 640

    # Set Dataloader
    vid_path, vid_writer = None, None
    dataset = LoadImages(source, img_size=imgsz, batch_size=BATCH_SIZE, skip_frames=skipLimit, stride=stride, multithreaded=False)

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

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    # Get required informations
    vid_fps = dataset.cap.get(cv2.CAP_PROP_FPS)
    vid_width = int(dataset.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_height = int(dataset.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    t3 = None
    prev_collisions = {}
    intval = 1
    for path, im0sb in dataset:
        # Inference
        t1 = time_synchronized()

        if t3 is not None:
            dataloader_time = (t1 - t3) / BATCH_SIZE * 1000
        else:
            dataloader_time = 0.0

        # Inference + NMS (YOLOX)
        preds, img_infos = predictor.inference_batched(im0sb)
        t2 = time_synchronized()

        # YOLOX postprocessing
        for b in range(BATCH_SIZE):
            det = preds[b]
            img_info = img_infos[b]
            im0s = im0sb[b]

            cls_conf = predictor.confthre
            ratio = img_info['ratio']

            # Process detections
            save_path = str(Path(out) / Path(path).name)

            xywhs = []
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

                if conf >= cls_conf:
                    img_h, img_w, _ = im0s.shape  # get image shape
                    x_c, y_c, bbox_w, bbox_h = main.bbox_rel(img_w, img_h, *[x0, y0, x1, y1])
                    obj = [x_c, y_c, bbox_w, bbox_h]

                    xywhs.append(obj)
                    confs.append([conf.item()])
                    clses.append([cls.item()])
                
            xywhs = torch.Tensor(xywhs)
            confs = torch.Tensor(confs)
            clses = torch.Tensor(clses)
            # Pass detections to deepsort
            outputs = []

            if not 'disable' in groundtruths_path:
                groundtruths = solution.load_labels(groundtruths_path, img_w,img_h, frame_num)

                if (groundtruths.shape[0]==0):
                    outputs = deepsort.update(xywhs, confs, clses, im0s)
                else:
                    # print(groundtruths)
                    xywhs = groundtruths[:,2:]
                    tensor = torch.tensor((), dtype=torch.int32)
                    confs = tensor.new_ones((groundtruths.shape[0], 1))
                    clses = groundtruths[:,0:1]
                    outputs = deepsort.update(xywhs, confs, clses, im0s)
                
                if frame_num >= 2:
                    for real_ID in groundtruths[:,1:].tolist():
                        for DS_ID in xyxy2xywh(outputs[:, :5]):
                            bx1, by1, bx2, by2, identity = DS_ID
                            r_identity, rbx1, rby1, rbx2, rby2 = real_ID
                            
                            # If BBox is same as GT, assign that ID to ID map
                            if (abs(bx1-rbx1)/img_w < 0.005) and (abs(by1-rby1)/img_h < 0.005) and (abs(bx2-rbx2)/img_w < 0.005) and(abs(by2-rby2)/img_h < 0.005):
                                # print("New Identity %s inserted as %d" % (identity, r_identity))
                                id_mapping[identity] = int(r_identity)
            else:
                outputs = deepsort.update(xywhs, confs, clses, im0s)

            # draw boxes for visualization
            if len(outputs) > 0:
                bbox_xyxy = outputs[:, :4]
                identities = outputs[:, 4]
                clses = outputs[:, 5]
                scores = outputs[:, 6]

                # Clip bbox_xyxy - fix for "Out of bound of Image Error"
                for i in range(bbox_xyxy.shape[0]):
                    x1, y1, x2, y2 = bbox_xyxy[i]
                    if vid_width <= x1 or vid_width <= x2 or \
                        vid_height <= y1 or vid_height <= y2:
                        bbox_xyxy[i] = np.array([0, 0, 0, 0])
                
                #Temp solution to get correct id's 
                mapped_id_list = []
                unknown_id_list = []
                for ids in identities:
                    if(ids in id_mapping):
                        mapped_id_list.append(int(id_mapping[ids]))
                    else:
                        #! TODO Reqired polishing of IDs!
                        # previous person ID must have loopup to original person ID
                        mapped_id_list.append(ids)

                # print("Not on Mapped ID list: ", unknown_id_list)

                collisions, bbox_strings, frame_catch_pairs, ball_person_pairs = solution.detect_catches(im0s, bbox_xyxy, clses, mapped_id_list, frame_num, colorDict, frame_catch_pairs, ball_person_pairs, colorOrder, prev_collisions, skipLimit, save_img)
                prev_collisions = collisions

                if save_img or view_img:
                    main.draw_boxes(im0s, bbox_xyxy, [names[i] if i < len(names) else 'Unknown' for i in clses], scores, bbox_strings, id_mapping, identities)
            
            # Stream results
            if view_img:
                cv2.imshow(path, cv2.resize(im0s, [1920, 1080]))
                ret = cv2.waitKey(intval)
                if ret == ord('q'):  # q to quit
                    raise StopIteration
                elif ret == ord('r'):  # r to run continuously
                    intval = 1
                elif ret == ord('p'): # p to pause
                    intval = 0

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0s)
                else:
                    #Draw frame number
                    tmp = framestr.format(frame = frame_num)
                    t_size = cv2.getTextSize(tmp, cv2.FONT_HERSHEY_PLAIN, 2 , 2)[0]
                    cv2.putText(im0s, tmp, (0, (t_size[1] + 10)), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)

                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*opt.fourcc), vid_fps, (vid_width, vid_height))
                    vid_writer.write(im0s)
            if frame_num > 10:
                frame_num += skipLimit
            frame_num += 1
        
        t3 = time_synchronized()
        
        #Inference Time
        forward_ms = (t2 - t1) / BATCH_SIZE * 1000
        postprocess_ms = (t3 - t2) / BATCH_SIZE * 1000
        vid_fps = ((BATCH_SIZE * (1+skipLimit))/(t3 - t1)) if frame_num > 10 else BATCH_SIZE/(t3 - t1)
        fpses.append(vid_fps)
        print('LSTCATCH=%s BATCH=%d DATA=%.2f ms/F MODL=%.2f ms/F POST=%.2f ms/F ALL=%.2f fps' % (
            "None" if len(frame_catch_pairs) == 0 else "\"Frame=%d/PersonID=%s\"" % (frame_catch_pairs[-1][0], frame_catch_pairs[-1][1].strip()),
            BATCH_SIZE, dataloader_time, forward_ms, postprocess_ms, vid_fps
            ))
        
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
