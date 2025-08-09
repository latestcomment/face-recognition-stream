import cv2

from models.face_detector.scrfd import SCRFD

class FaceDetection():
    def __init__(self, detector, detector_model):
        self.detector_model = detector_model

        if detector == "SCRFD":
            self.detector = SCRFD(model_file=self.detector_model)
            self.detector.prepare(-1) 

    def detect(self, image):

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        bboxes, _ = self.detector.detect(image, 0.5, input_size = (320, 320))

        if len(bboxes) < 1:
            face_bboxes = None
        else:
            face_bboxes = bboxes
        return face_bboxes
    
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