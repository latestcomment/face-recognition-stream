import onnxruntime as ort

import cv2
import numpy as np

class FaceAntiSpoofing():
    def __init__(self, spoofing_classifier_model):
        self.spoofing_classifier_model = spoofing_classifier_model
        self.session = ort.InferenceSession(self.spoofing_classifier_model)

    def detect_spoofing(self, frame):
        img = preprocessed_image(frame)
        ort_inputs = {self.session.get_inputs()[0].name: img}
        pred = self.session.run(None, ort_inputs)[0][0]
        prob = softmax(pred)
        return prob

def preprocessed_image(frame):
    img = cv2.resize(frame, (224, 224))
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img - mean) / std

    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0).astype(np.float32)
    return img

def softmax(pred):
    """Compute softmax values for each sets of scores in prediction."""
    e_x = np.exp(pred - np.max(pred))
    return e_x / e_x.sum(axis=0)