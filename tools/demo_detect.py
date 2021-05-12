
import sys
sys.path.insert(0, '.')
import argparse
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import cv2
import time

import lib.transform_cv2 as T
from lib.models import model_factory
from configs import cfg_factory

torch.set_grad_enabled(False)
np.random.seed(123)


# args
parse = argparse.ArgumentParser()
parse.add_argument('--model', dest='model', type=str, default='bisenetv2')
parse.add_argument('--weight-path', type=str, default='./res/model_final.pth')
parse.add_argument('--source', help='Video path', type=str, default='./output.mp4')
parse.add_argument('--show', type=bool, default=False)
args = parse.parse_args()
cfg = cfg_factory[args.model]


palette = np.random.randint(0, 256, (256, 3), dtype=np.uint8)

# define model
net = model_factory[cfg.model_type](19)
net.load_state_dict(torch.load(args.weight_path, map_location='cpu'))
net.eval()
net.cuda()

cap = cv2.VideoCapture(args.source)
    
times_infer, times_pipe = [], []

# prepare data
to_tensor = T.ToTensor(
    mean=(0.3257, 0.3690, 0.3223), # city, rgb
    std=(0.2112, 0.2148, 0.2115),
)
while True:
    _, frame = cap.read()
    frame_hor_scaled = cv2.resize(frame, (1024, frame.shape[0]), interpolation = cv2.INTER_AREA)
    frame_scaled = cv2.resize(frame_hor_scaled, (frame_hor_scaled.shape[1], 512), interpolation = cv2.INTER_AREA)
    im = to_tensor(dict(im=frame_scaled, lb=None))['im'].unsqueeze(0).cuda()

    t0 = time.time()
    # inference
    out = net(im)[0].argmax(dim=1).squeeze().detach().cpu().numpy()
    pred = palette[out]
    overlayed = 0.7 * frame_scaled + 0.3 * pred
    overlayed = overlayed.astype('uint8')

    t1 = time.time()
    t2 = time.time()

    times_infer.append(t1-t0)
    times_pipe.append(t2-t0)
            
    times_infer = times_infer[-20:]
    times_pipe = times_pipe[-20:]

    ms = sum(times_infer)/len(times_infer)*1000
    fps_infer = 1000 / (ms+0.00001)
    fps_pipe = 1000 / (sum(times_pipe)/len(times_pipe)*1000)
            
    overlayed = cv2.putText(overlayed, "Time: {:.1f}FPS".format(fps_infer), (0, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
    if args.show:
        cv2.imshow('Segmented ' + args.model , overlayed)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break            
    print("Time: {:.2f}ms, Detection FPS: {:.1f}, total FPS: {:.1f}".format(ms, fps_infer, fps_pipe))
    
cap.release()
