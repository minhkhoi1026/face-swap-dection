import os
import shutil

def face_to_landmark_file(file):
    return os.path.splitext(file)[0] + "_landmark.png"


for data in ["test","train"]:
    for label in ["fake","real"]:
        landmark_src = "fsd-deepfakerapp/dfapp_plus_landmark/landmark/{}".format(label)
        landmark_dst = "fsd-deepfakerapp/dfapp_plus_landmark/{}/landmark/{}".format(data,label)
        face_src = "fsd-deepfakerapp/dfapp_plus/{}/{}".format(data,label)
        face_dst = "fsd-deepfakerapp/dfapp_plus_landmark/{}/face/{}".format(data,label)

        landmark_list = os.listdir(landmark_src)
        for face in os.listdir(face_src):
            landmark = face_to_landmark_file(face)
            if landmark in landmark_list:
                shutil.copy(os.path.join(face_src,face), os.path.join(face_dst,face))
                shutil.move(os.path.join(landmark_src,landmark), os.path.join(landmark_dst,landmark))
            else:
                print(face)
            
