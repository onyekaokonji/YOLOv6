import os, requests, torch, math, cv2
import numpy as np
import PIL
import sys
# import subprocess

from yolov6.utils.events import LOGGER, load_yaml
from yolov6.layers.common import DetectBackend
from yolov6.data.data_augment import letterbox
from yolov6.utils.nms import non_max_suppression
from yolov6.core.inferer import Inferer
from tools.infer import run

import streamlit as st

st.set_page_config(page_title='Drug Counter', layout = 'centered', initial_sidebar_state='auto')

st.title("Detect and Count Objects")    

def videoInput():
    uploaded_video = st.file_uploader("Upload Video", type=['mp4', 'mpeg', 'mov'])
    if uploaded_video != None:

        vidpath = os.path.join('data/videos', uploaded_video.name)
        outputpath = os.path.join('runs', 'inference')
        cfg_model_path = os.path.join('assets/weights', 'yolov6n.pt')

        with open(vidpath, mode='wb') as f:
            f.write(uploaded_video.read())  # save video to disk

        st_video = open(vidpath, 'rb')
        video_bytes = st_video.read()
        st.video(video_bytes)
        st.caption("Uploaded Video")
        st.write("Please wait while the objects are detected")
        run(weights=cfg_model_path, source=vidpath, yaml=os.path.join('data', 'coco.yaml'), img_size=640, half=False,
            conf_thres=0.4, iou_thres=0.45, max_det=1000, save_txt=False, not_save_img=False, save_dir=None, view_img=False,
            classes=None, agnostic_nms=False, project=outputpath, hide_labels=False, hide_conf=False)
        final_output_path = os.path.join(outputpath, 'exp/') + uploaded_video.name
        st_video2 = open(final_output_path, 'rb')
        video_bytes2 = st_video2.read()
        st.video(video_bytes2)
        st.caption("Model Prediction")

def main():
    # -- Sidebar
    # st.sidebar.title('Options')
            
    option = st.sidebar.radio("Select input type.", ['Video', 'Image'])

    st.subheader('👈🏽 Select options left-haned menu bar.')
    if option == "Video": 
        videoInput()

if __name__ == '__main__':
    main()