import os
import cv2
import json
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

src = "./data_extract/deepfaker_app/sample"
dst = "./data_extract/deepfaker_app/face_landmarks"
vis = "./data_extract/deepfaker_app/face_landmarks_vis"


drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
def visualize(annotated_image,face_landmarks):
  mp_drawing.draw_landmarks(
      image=annotated_image,
      landmark_list=face_landmarks,
      connections=mp_face_mesh.FACEMESH_TESSELATION,
      landmark_drawing_spec=None,
      connection_drawing_spec=mp_drawing_styles
      .get_default_face_mesh_tesselation_style())
  mp_drawing.draw_landmarks(
      image=annotated_image,
      landmark_list=face_landmarks,
      connections=mp_face_mesh.FACEMESH_CONTOURS,
      landmark_drawing_spec=None,
      connection_drawing_spec=mp_drawing_styles
      .get_default_face_mesh_contours_style())
  mp_drawing.draw_landmarks(
      image=annotated_image,
      landmark_list=face_landmarks,
      connections=mp_face_mesh.FACEMESH_IRISES,
      landmark_drawing_spec=None,
      connection_drawing_spec=mp_drawing_styles
      .get_default_face_mesh_iris_connections_style())
  cv2.imwrite("{}_landmarks_vis.png".format(os.path.join(vis,dir,name)), annotated_image)


def write_data(face_landmarks):
  landmarks_data = list()
  for data_point in face_landmarks.landmark:
    landmarks_data.append({"x":data_point.x,"y":data_point.y,"z":data_point.z})
  json_object = json.dumps(landmarks_data, indent=4)
  with open("{}_landmark.json".format(os.path.join(dst,dir,name)), "w") as outfile:
    outfile.write(json_object)


with mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5) as face_mesh:
  for dir in os.listdir(src):
    for file in os.listdir(os.path.join(src,dir)):
      name, ext = os.path.splitext(file)
      image = cv2.imread(os.path.join(src,dir,file))
      # Convert the BGR image to RGB before processing.
      results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

      # Print and draw face mesh landmarks on the image.
      # for face_landmarks in results.multi_face_landmarks:
      if not results.multi_face_landmarks:
        print("No face in {}".format(os.path.join(dir,file)))
        continue
      face_landmarks = results.multi_face_landmarks[0]

      write_data(face_landmarks)
        
      visualize(image,face_landmarks)