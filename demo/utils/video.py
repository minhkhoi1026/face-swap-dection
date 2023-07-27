import cv2

def get_video_frame_rate(video_file):
    # Open the video file
    cap = cv2.VideoCapture(video_file)

    # Get the frames per second (fps) from the video properties
    frame_rate = cap.get(cv2.CAP_PROP_FPS)

    # Release the video capture object
    cap.release()

    return frame_rate
