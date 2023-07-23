from demo.extractor.base_extractor import BaseExtractor
import cv2
import numpy as np
import os
import typing
import face_alignment

class FaceFAFIExtractor(BaseExtractor):
    def __init__(self, thickness_percentage, blur_percentage):
        self.thickness_percentage = thickness_percentage
        self.blur_percentage = blur_percentage
        self.face_landmark_predictor = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)
        self.pred_types = {'face': slice(0, 17),
                    'eyebrow1': slice(17, 22),
                    'eyebrow2': slice(22, 27),
                    'nose': slice(27, 31),
                    'nostril': slice(31, 36),
                    'eye1': slice(36, 42),
                    'eye2': slice(42, 48),
                    'lips': slice(48, 60),
                    'teeth': slice(60, 68),
                    }
    
    def extract(self, data_path: typing.Union[str, os.PathLike]):
        image = cv2.imread(data_path)
        dest_folder = os.path.dirname(data_path)
        file_prefix = os.path.splitext(os.path.basename(data_path))[0]

        landmark_pred = self.face_landmark_predictor.get_landmarks(image)
        
        if not landmark_pred:
            return []
        
        face_landmark = []

        for id, face in enumerate(landmark_pred):
            landmark_vis = np.zeros(image.shape, dtype=np.uint8)

            listY = [int(y) for x,y in face]
            face_size = max(listY) - min(listY)

            thickness = int(face_size * self.thickness_percentage / 100)
            blur = int(face_size * self.blur_percentage / 100)

            for key, value in self.pred_types.items():
                cur_landmarks = face[value].tolist()

                if key in ["lips", "eye1", "eye2"]:
                    cur_landmarks.append(cur_landmarks[0])
                for i in range(len(cur_landmarks)-1):
                    pt1 = (int(cur_landmarks[i][0]), int(cur_landmarks[i][1]))
                    pt2 = (int(cur_landmarks[i+1][0]), int(cur_landmarks[i+1][1]))

                    cv2.line(landmark_vis, pt1, pt2, (255, 255, 255), thickness)
            blurred_img = cv2.blur(landmark_vis, (blur, blur))
            scaled_image = blurred_img / 255
            result_image = image * scaled_image

            non_zero_pixels = np.nonzero(result_image)
            min_y = np.min(non_zero_pixels[0])
            max_y = np.max(non_zero_pixels[0])
            min_x = np.min(non_zero_pixels[1])
            max_x = np.max(non_zero_pixels[1])

            landmark_crop = result_image[min_y:max_y+1, min_x:max_x+1]
            face_crop = image[min_y:max_y+1, min_x:max_x+1]

            landmark_path = os.path.join(dest_folder, "{}_{:02d}_landmark.png".format(file_prefix, id))
            face_path = os.path.join(dest_folder, "{}_{:02d}_face.png".format(file_prefix, id))
            
            cv2.imwrite(landmark_path, landmark_crop)
            cv2.imwrite(face_path, face_crop)

            face_landmark.append({"face_path": face_path, "fafi_path": landmark_path, "bbox":(min_x,min_y,max_x,max_y)})

        return face_landmark
