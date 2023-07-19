from demo.extractor.base_extractor import BaseExtractor
import cv2
import tempfile
from tqdm import tqdm
import os
import typing

class FrameExtractor(BaseExtractor):
    def __init__(self, frame_width, frame_height):
        self.frame_width = frame_width
        self.frame_height = frame_height
    
    def extract(self, data_path: typing.Union[str, os.PathLike], sampling: int):
        """
        Method to extract frames from video using opencv2

        Args:
            data_path (PathLike): path to original video
            sampling (int): samping

        Returns:
            _type_: _description_
        """
        dest_path = tempfile.mkdtemp()

        face_landmark = []

        reader = cv2.VideoCapture(data_path)

        width = int(reader.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # Calculate the aspect ratio
        aspect_ratio = width / height
        
        new_width = self.frame_width
        new_height = self.frame_height

        if aspect_ratio > new_width / new_height:
            new_height = int(new_width / aspect_ratio)
        else:
            new_width = int(new_height * aspect_ratio)
        if width > new_width:
            print(f"This video will be downscaled from ({width},{height}) to ({new_width},{new_height}).")

        total_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_paths = []
        
        for frame_id in tqdm(range(total_frames)):
            success = reader.grab()

            if frame_id % sampling:
                continue
            
            success, frame = reader.retrieve()
            
            if not success:
                continue

            if width > new_width:
                frame = cv2.resize(frame, (new_width, new_height))

            # extract faces from frame
            frame_path = os.path.join(dest_path, "{:06d}.png".format(frame_id))
            cv2.imwrite(frame_path, frame)
            frame_paths.append({"frame_id": frame_id, "frame_path": frame_path})

        reader.release()

        return frame_paths
