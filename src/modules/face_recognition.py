
import onnxruntime as ort

import cv2
import numpy as np

class FaceRecognition():
    def __init__(self, recognizer_model):
        self.recognizer_model = recognizer_model
        self.session = ort.InferenceSession(self.recognizer_model)

    def get_face_feature(self, face):
        face = preprocessed_face(face)
        ort_inputs = {self.session.get_inputs()[0].name: face}
        face_feature = self.session.run(None, ort_inputs)[0][0]
        return face_feature
    
    def get_similarity(self, face_feature, features):
        face_feature_norm = self._normalize_feature(face_feature)
        features_norm = features / np.linalg.norm(features)
        similarities = features_norm @ face_feature_norm

        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]

        return best_idx, best_score, similarities
    
    def _normalize_feature(self, face_feature):
        return face_feature / np.linalg.norm(face_feature)
    

def preprocessed_face(face):
    face = cv2.resize(face, (112, 112))
    face = np.transpose(face, (2, 0, 1))
    face = np.expand_dims(face, axis=0).astype(np.float32)
    return face