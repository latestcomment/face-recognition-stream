import onnxruntime as ort
import psycopg2

import cv2
import numpy as np

import os

from src.models.face_detector.scrfd import SCRFD

def detect_face(img, face_detector_model):
    detector = SCRFD(model_file=face_detector_model)
    detector.prepare(-1)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    bboxes, _ = detector.detect(img, 0.5, input_size = (320, 320))

    face = crop_face(img, bboxes[0])
    return face

def crop_face(image, bbox, margin=0):
    h, w, _ = image.shape
    x1, y1, x2, y2 = bbox[:4]
    
    # Add margin and clamp to image boundaries
    x1 = max(int(x1 - margin), 0)
    y1 = max(int(y1 - margin), 0)
    x2 = min(int(x2 + margin), w)
    y2 = min(int(y2 + margin), h)
    
    cropped_face = image[y1:y2, x1:x2]
    return cropped_face

def preprocessed_face(face):
    face = cv2.resize(face, (112, 112))
    face = np.transpose(face, (2, 0, 1))
    face = np.expand_dims(face, axis=0).astype(np.float32)
    return face

def insert_data_to_postgres(data):
    try:
        conn = psycopg2.connect(
            database="dev", user='latestcomment',
            password='dev123', host='localhost', port='5432')
        
        conn.autocommit = False
        cursor = conn.cursor()

        query = """
            INSERT INTO person (person_name, embedding) VALUES (%s, %s)
        """
        args = data['name'], data['embedding'].tolist()
        cursor.execute(query, args)

        conn.commit()

    except (Exception, psycopg2.DatabaseError) as error:
        print("Error in transaction, reverting all changes using rollback ", error)
        conn.rollback()

    finally:
        if conn:
            cursor.close()
            conn.close()

def main():
    model_face_detector = 'src/models/scrfd_500m.onnx'
    session = ort.InferenceSession("src/models/adaface-ir_18.onnx")

    dataset_path = "images/"

    name_dict = {}
    features = []

    dirs = [e for e in os.scandir(dataset_path) if e.is_dir()]

    for idx, entry in enumerate(dirs):
        files = os.listdir(entry.path)
        file_path = os.path.join(entry.path, files[0])

        img = cv2.imread(file_path)
        face = detect_face(img, model_face_detector)

        face = preprocessed_face(face)

        ort_inputs = {session.get_inputs()[0].name: face}
        pred = session.run(None, ort_inputs)[0][0]

        features.append(pred)
        name_dict[idx] = entry.name

        data = {"name": entry.name, "embedding": pred}
        insert_data_to_postgres(data)

    features = np.array(features)
    np.savez("data/dataset_features.npz", features=features, name_dict=name_dict)

if __name__=="__main__":
    main()