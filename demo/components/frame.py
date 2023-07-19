import streamlit as st
import cv2
from demo.utils.gradcam import show_cam_on_image

@st.cache_data
def visualize_frame_prediction(frame_data, fake_label, is_show_gradcam=False):
    image = cv2.imread(frame_data["frame_path"])
    for pred in frame_data["predict"]:
        x, y, p, q = pred["bbox"]
        score = pred["score"]
        
        if is_show_gradcam:
            gradcam = pred["grad_cam"]
            bbox = (x, y, p, q)
            image = show_cam_on_image(image / 255, gradcam, bbox)
        
        # Determine the color based on the score
        colors = [(0, 255, 0), # Green 
                  (0, 0, 255) # Red
                ]
        color_id = fake_label & (0 if score < 50 else 1)
        color = colors[color_id]
            
        # Draw the bbox on the image
        cv2.rectangle(image, (x, y), (p, q), color, 2)
        # Put the score text on top of the bbox
        text = f'{score:.2f}'
        text_position = (x, y - 10)
        cv2.putText(image, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image
