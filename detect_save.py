import argparse
import os
import platform
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, plot_one_box, strip_optimizer)
from utils.torch_utils import select_device, load_classifier, time_synchronized

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


class detecter():
    def __init__(self, weights, img_size=640, conf_thres=0.5,save_img=False):
        self.weights = weights
        self.imgsz = img_size
        self.conf_thres = conf_thres
        self.out = 'inference/output'
        self.iou_thres = 0.5
        self.device = ''
        self.view_img = False
        self.save_txt = False

        # webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

        # Initialize
        self.device = select_device(self.device)
        if os.path.exists(self.out):
            shutil.rmtree(self.out)  # delete output folder
        os.makedirs(self.out)  # make new output folder
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        self.model = attempt_load(self.weights, map_location=self.device)  # load FP32 model
        self.imgsz = check_img_size(self.imgsz, s=self.model.stride.max())  # check img_size
        if self.half:
            self.model.half()  # to FP16

        # Second-stage classifier
        self.classify = False
        if self.classify:
            modelc = load_classifier(name='resnet101', n=2)  # initialize
            modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=self.device)['model'])  # load weights
            modelc.to(self.device).eval()

    def detect(self, img0_path):
        with torch.no_grad():
            # Set Dataloader
            vid_path, vid_writer = None, None
            save_img = True
            # dataset = LoadImages(source, img_size=self.imgsz)

            # Get names and colors
            names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
            colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

            # Run inference
            t0 = time.time()
            # img = torch.from_numpy(img).to(device)
            # img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
            # _ = self.model(img.half() if half else img) if device.type != 'cpu' else None  # run once
            # Padded resize
            img0 = cv2.imread(img0_path)
            img = letterbox(img0, new_shape=self.imgsz)[0]

            # Convert
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)

            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = time_synchronized()
            pred = self.model(img)[0]

            # Apply NMS
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)
            t2 = time_synchronized()

            # Apply Classifier (# Second-stage classifier)
            if self.classify:
                pred = apply_classifier(pred, modelc, img, img0)

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                p, s, im0 = img0_path, '', img0

                save_path = str(Path(self.out) / Path(p).name)
                txt_path = str(Path(self.out) / Path(p).stem)
                s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += '%g %ss, ' % (n, names[int(c)])  # add to string

                    # Write results
                    for *xyxy, conf, cls in det:
                        if save_img or view_img:  # Add bbox to image
                            label = '%s %.2f' % (names[int(cls)], conf)
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

                # Print time (inference + NMS)
                print('%sDone. (%.3fs)' % (s, t2 - t1))

                # Save results (image with detections)
                if save_img:
                    cv2.imwrite(save_path, im0)
                    # else:
                    #     if vid_path != save_path:  # new video
                    #         vid_path = save_path
                    #         if isinstance(vid_writer, cv2.VideoWriter):
                    #             vid_writer.release()  # release previous video writer

                    #         fourcc = 'mp4v'  # output video codec
                    #         fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    #         w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    #         h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    #         vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    #     vid_writer.write(im0)

            # if save_img:
            #     print('Results saved to %s' % Path(out))
            #     if platform == 'darwin' and not opt.update:  # MacOS
            #         os.system('open ' + save_path)

            print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    d = detecter(weights="runs/exp1/weights/best.pt")
    d.detect('inference/images/2020_04_16 09.22.23.537(MP0903013-130)A.jpg')
