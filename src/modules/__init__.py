import os

from db.postgres import PostgresDB
from conf.config import load_config
from modules.face_detection import FaceDetection
from modules.face_recognition import FaceRecognition
from modules.face_anti_spoofing import FaceAntiSpoofing

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_CONF_PATH = os.path.join(BASE_DIR, "..", "conf", "db.conf")
APP_CONF_PATH = os.path.join(BASE_DIR, "..", "conf", "app.conf")

# DATABASE
pg_config = load_config(DB_CONF_PATH, "POSTGRES")
vector_db = PostgresDB(pg_config)

# CAMERA
camera_config = load_config(APP_CONF_PATH, "CAMERA")

# APP
face_detection_config = load_config(APP_CONF_PATH, "FACE DETECTION")
face_anti_spoofing_config = load_config(APP_CONF_PATH, "FACE ANTI SPOOFING")
face_recognition_config = load_config(APP_CONF_PATH, "FACE RECOGNITION")

detector = FaceDetection(detector="SCRFD", detector_model=os.path.join(BASE_DIR, "..", "models", face_detection_config.get("MODEL")))
spoofing_classifier = FaceAntiSpoofing(spoofing_classifier_model=os.path.join(BASE_DIR, "..", "models", face_anti_spoofing_config.get("MODEL")))
recognizer = FaceRecognition(recognizer_model=os.path.join(BASE_DIR, "..", "models", face_recognition_config.get("MODEL")))