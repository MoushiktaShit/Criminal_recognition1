import os
import yaml
from dotenv import load_dotenv


# ==========================================
# LOAD ENV
# ==========================================
load_dotenv()


# ==========================================
# LOAD YAML SETTINGS
# ==========================================
BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

SETTINGS_PATH = os.path.join(
    BASE_DIR,
    "utils",
    "settings.yaml"
)

with open(SETTINGS_PATH, "r") as file:
    settings = yaml.safe_load(file)


# ==========================================
# DATABASE CONFIG
# ==========================================
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DB_PORT = os.getenv("DB_PORT")


# ==========================================
# EMAIL CONFIG
# ==========================================
MAIL_SENDER = os.getenv("MAIL_SENDER")
MAIL_PASSWORD = os.getenv("MAIL_PASSWORD")
MAIL_RECEIVER = os.getenv("MAIL_RECEIVER")


# ==========================================
# STATION DETAILS
# ==========================================
STATION_NAME = settings["station"]["name"]
PLATFORM_NUMBER = settings["station"]["platform"]


# ==========================================
# PATHS
# ==========================================
VIDEO_PATH = settings["paths"]["video_path"]

OUTPUT_PATH = settings["paths"]["output_path"]

INSIGHTFACE_ROOT = settings["paths"]["insightface_root"]


# ==========================================
# MODELS
# ==========================================
YOLO_MODEL = settings["models"]["yolo_model"]

TRACKER = settings["models"]["tracker"]


# ==========================================
# EVIDENCE FOLDERS
# ==========================================
FACE_EVIDENCE_DIR = settings["evidence"]["face_dir"]

FRAME_EVIDENCE_DIR = settings["evidence"]["frame_dir"]

GROUP_EVIDENCE_DIR = settings["evidence"]["group_dir"]


# ==========================================
# CREATE EVIDENCE FOLDERS
# ==========================================
os.makedirs(FACE_EVIDENCE_DIR, exist_ok=True)

os.makedirs(FRAME_EVIDENCE_DIR, exist_ok=True)

os.makedirs(GROUP_EVIDENCE_DIR, exist_ok=True)


# ==========================================
# THRESHOLDS
# ==========================================
FRAME_SKIP = settings["thresholds"]["frame_skip"]

HIGH_CONF_THRESHOLD = settings["thresholds"]["high_confidence"]

LOW_CONF_THRESHOLD = settings["thresholds"]["low_confidence"]

MIN_SCORE_GAP = settings["thresholds"]["min_score_gap"]

MIN_FACE_AREA = settings["thresholds"]["min_face_area"]

MIN_DET_SCORE = settings["thresholds"]["min_det_score"]

FACE_OVERLAP_MIN = settings["thresholds"]["face_overlap_min"]

FACE_ZONE_FRACTION = settings["thresholds"]["face_zone_fraction"]

TRACK_LOST_BUFFER = settings["thresholds"]["track_lost_buffer"]

CONFIRM_FRAMES_NEEDED = settings["thresholds"]["confirm_frames_needed"]

CONFIRM_WINDOW = settings["thresholds"]["confirm_window"]

VOTE_TTL = settings["thresholds"]["vote_ttl"]


# ==========================================
# KNOWN CRIMINALS
# ==========================================
KNOWN_CRIMINALS = set(
    settings["criminals"]
)