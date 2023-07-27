import streamlit as st
import tempfile
import os
from demo.utils.visualize import visualize_frame_prediction, create_video_from_frames

@st.cache_data
def visualize_frame_prediction_st(frame_data, fake_label, is_show_gradcam=False):
    return visualize_frame_prediction(frame_data, fake_label, is_show_gradcam)

@st.cache_data
def visualize_video_prediction(video_data, fake_label, frame_rate, is_show_gradcam=False):
    frames = []
    for _, frame_data in video_data.iterrows():
        frames.append(visualize_frame_prediction_st(frame_data, fake_label, is_show_gradcam))
    return create_video_from_frames(frames, frame_rate)
