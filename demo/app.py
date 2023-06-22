import streamlit as st
import pandas as pd
import altair as alt
import typing
import numpy as np
import cv2
import tempfile
import os
from components.chart import (
    create_overall_result_chart,
    create_frame_detail_result_chart,
)
from components.frame import visualize_frame_prediction
from demo.detector import DETECTOR_REGISTRY

def get_predictions(video_bytes: bytes, selected_model: typing.List[str]) -> pd.DataFrame:
    result = []
    for model_name in selected_model:
        model = DETECTOR_REGISTRY.get(model_name)()
        result.append([model_name, model.predict(video_bytes)])
    return pd.DataFrame(result, columns=["model", "frame_predictions"])

def aggregate_predictions(predictions: list) -> float:
    """Aggreate mean prediction of all frame"""
    def aggregate_face_scores(face_data: list) -> float:
        """Aggregate faces score in one frame by taking the minimum score."""
        return np.min(list(x["score"] for x in face_data)) if face_data else 100.0
    return predictions.apply(lambda x: aggregate_face_scores(x)).mean().round(2)

def main():
    st.set_page_config(
        page_title="Anti Deepfake",
        page_icon="🕵️",
        layout="wide"
    )
    
    model_list = DETECTOR_REGISTRY.list_all()
    st.title("Deepfake Detection Tool")
    
    # create an form
    input_form = st.form("form")
    input_form.write("Upload a video to detect if it is a deepfake video.")

    uploaded_file = input_form.file_uploader("video_uploader", type="mp4")
    selected_model = input_form.multiselect("Choose all detector you want", model_list)
    
    submitted = input_form.form_submit_button("Submit")
    
    if submitted:
        if uploaded_file and selected_model:
            st.success("Your video is uploaded successfully.")
        else:
            st.error("Please upload a video and choose at least one model.")
    
    if uploaded_file and selected_model:
        # get prediction data
        video_bytes = uploaded_file.read()
        model_predictions = get_predictions(video_bytes, selected_model)
        
        # overall result
        st.subheader("Your video overall result")
        overall_prediction = pd.DataFrame({
            "model": model_predictions["model"],
            "predict": model_predictions["frame_predictions"].apply(lambda x: aggregate_predictions(x["predict"]))
        })
        st.altair_chart(create_overall_result_chart(overall_prediction), use_container_width=True)

        # inspect detail result
        st.subheader("Detail investigation")
        choose_model = st.selectbox("Choose a model you want to investigate", selected_model, index=0)
        
        if choose_model:
            frame_data = model_predictions[model_predictions["model"] == choose_model]["frame_predictions"].iloc[0]
        
            # area chart
            point_df = frame_data[["frame_id"]]
            point_df["predict"] = frame_data["predict"].apply(lambda x: min(item["score"] for item in x) if x else 100.0)
            
            st.altair_chart(create_frame_detail_result_chart(point_df), use_container_width=True)

            # frame visualization
            frame_idx = st.slider("Frame", 0, len(frame_data) - 1)
            frame = visualize_frame_prediction(frame_data.iloc[frame_idx])
            st.image(frame, channels="BGR", use_column_width=True)
        


if __name__ == "__main__":
    main()
