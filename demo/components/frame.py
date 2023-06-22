import streamlit as st
import cv2

@st.cache_data
def visualize_frame_prediction(frame_data):
    image = cv2.imread(frame_data["frame_path"])
    for pred in frame_data["predict"]:
        x, y, p, q = pred["bbox"]
        score = pred["score"]
        # Determine the color based on the score
        if score >= 50:
            color = (0, 255, 0)  # Green
        else:
            color = (0, 0, 255)  # Red
        # Draw the bbox on the image
        cv2.rectangle(image, (x, y), (p, q), color, 2)
        # Put the score text on top of the bbox
        text = f'{score:.2f}'
        text_position = (x, y - 10)
        cv2.putText(image, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image
