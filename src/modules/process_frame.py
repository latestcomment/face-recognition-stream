import cv2

from modules.face_detection import crop_face
from modules.draw import draw_face

from . import (
    vector_db,
    face_recognition_config, face_anti_spoofing_config,
    detector, recognizer, spoofing_classifier
)

def process_frame(frame):
    # detect face
    face_bboxes = detector.detect(frame)
    if face_bboxes is None:
        return frame

    for face_bbox in face_bboxes:
        face = crop_face(frame, face_bbox)
        
        # check spoofing
        spoof_prob = spoofing_classifier.detect_spoofing(face)

        if spoof_prob[1] >= face_anti_spoofing_config.get("SPOOF_THRESHOLD"):
            liveness_status = 'spoof'
        else:
            liveness_status = 'real'

        # recognize face
        face_feature = recognizer.get_face_feature(face)

        result = vector_db.get_similarity(face_feature, limit=1)
        person_id, person_name, score = result[0]

        if score < face_recognition_config.get("SIMILARITY_THRESHOLD"):
            person_name = "not detected"
            score = 0.0
        
        # draw annotation in frame
        annotated_frame = draw_face(frame, face_bbox[:4], person_name, score, liveness_status, spoof_prob[1])

    return annotated_frame