import cv2
import tempfile
from demo.utils.gradcam import show_cam_on_image

def visualize_frame_prediction(frame_data, fake_label, is_show_gradcam=False):
    image = cv2.imread(frame_data["frame_path"])
    for pred in frame_data["predict"]:
        x, y, p, q = pred["bbox"]
        score = pred["score"]
        
        if is_show_gradcam:
            gradcam = pred["grad_cam"] # * score / 100
            bbox = (x, y, p, q)
            image = show_cam_on_image(image / 255, gradcam, bbox)
        
        # Determine the color based on the score
        colors = [(0, 255, 0), # Green 
                  (0, 0, 255) # Red
                  (0, 255, 255) # Yellow
                ]
        color_id = fake_label & (0 if score < 40 else 1 if score > 60 else 2)
        color = colors[color_id]
            
        # Draw the bbox on the image
        cv2.rectangle(image, (x, y), (p, q), color, 2)
        height, _, _ = image.shape
        # Put the score text on top of the bbox
        text = f'{score:.2f}'
        text_position = (x, y - 10) if y > 10 else (x, y + q) if y + q + 10 > height else (x + 3,y)
        cv2.putText(image, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image

def create_video_from_frames(frames):
    frame_rate = 30
    save_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    # Define the video codec and create a VideoWriter object
    codec = cv2.VideoWriter_fourcc(*'mp4v')
    # Call the function to get the image dimensions
    width, height, _ = frames[0].shape
    print(width, height)
    out = cv2.VideoWriter(save_path.name, codec, frame_rate, (height, width))
    
    # Loop through the frames and write each frame to the video writer
    for frame in frames:
        # Resize the frame to match the desired size (if needed)
        frame = cv2.resize(frame, (height, width))

        out.write(frame)

    # Release the video writer and destroy any remaining windows
    out.release()
    cv2.destroyAllWindows()
    
    return save_path.name
