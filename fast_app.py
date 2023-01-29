import os, requests, torch, math, cv2
import numpy as np
import PIL
import sys
# import subprocess

from yolov6.utils.events import LOGGER, load_yaml
from yolov6.layers.common import DetectBackend
from yolov6.data.data_augment import letterbox
from yolov6.utils.nms import non_max_suppression
# from yolov6.core.inferer import Inferer
from tools.infer import run

from fastapi import FastAPI, UploadFile
from fastapi.responses import StreamingResponse

import shutil

app = FastAPI()

@app.post('/upload_video')
async def get_predictions(file: UploadFile):

    vidpath = os.path.join('data/videos', file.filename)
    outputpath = os.path.join('runs', 'inference')
    cfg_model_path = os.path.join('assets/weights', 'train_model.pt')
    print("File name: " + str(file.filename))

    with open (os.path.join('data/videos', f'{file.filename}'), 'wb') as f:
        shutil.copyfileobj(file.file, f)

    run(weights=cfg_model_path, source=vidpath, yaml=os.path.join('data', 'dataset.yaml'), img_size=640, half=False,
        conf_thres=0.4, iou_thres=0.45, max_det=1000, save_txt=False, not_save_img=False, save_dir=None, view_img=True,
        classes=None, agnostic_nms=False, project=outputpath, hide_labels=False, hide_conf=False)
    
    final_output_path = os.path.join(outputpath, 'exp', f'{file.filename}')

    def iterfile():
        with open (final_output_path, mode="rb") as g:
            yield from g
    
    return StreamingResponse(iterfile(), media_type="video/mp4")