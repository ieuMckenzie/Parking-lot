import os
import sys
import argparse
import glob
import time
import csv
import datetime
import re
import warnings 
import threading
import queue
import itertools 
from difflib import SequenceMatcher

import cv2
import numpy as np
import easyocr
from ultralytics import YOLO

# --- CONFIGURATION ---
warnings.filterwarnings("ignore") 

SIMILARITY_THRESH = 0.60  
OCR_THRESH = 0.80           
LOG_COOLDOWN = 60.0       
PLATE_REGEX = r'^[A-Z0-9]{5,8}$' 
AUTHORIZED_FILE = 'authorized.txt'
STABILITY_VOTES_REQUIRED = 10

# --- ARGUMENTS ---
parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Path to YOLO model file', required=True)
parser.add_argument('--source', help='Image source', required=True)
parser.add_argument('--thresh', help='Confidence threshold', default=0.5)
parser.add_argument('--resolution', help='Resolution WxH', default=None)
parser.add_argument('--record', help='Record video', action='store_true')
args = parser.parse_args()

model_path = args.model
img_source = args.source
min_thresh = float(args.thresh)
user_res = args.resolution
record = args.record

# --- INITIALIZATION ---
if not os.path.exists(model_path):
    sys.exit('ERROR: Model path is invalid.')

model = YOLO(model_path, task='detect')
labels = model.names

print("Initializing EasyOCR...")
reader = easyocr.Reader(['en'], gpu=True) 

log_filename = 'parking_log.csv'
seen_plates = {} 
plate_votes = {} 
authorized_plates = set() 
last_auth_update = 0

# --- TEXT FILE LOADING ---
def update_authorized_list():
    global last_auth_update, authorized_plates
    if time.time() - last_auth_update > 5:
        if os.path.exists(AUTHORIZED_FILE):
            try:
                with open(AUTHORIZED_FILE, 'r') as f:
                    new_set = {line.strip().upper() for line in f if line.strip()}
                if new_set != authorized_plates:
                    print(f"🔄 Reloaded Authorized List: {len(new_set)} plates.")
                    authorized_plates = new_set
            except Exception as e:
                print(f"Error reading text file: {e}")
        last_auth_update = time.time()

def is_authorized(plate):
    return plate in authorized_plates

# --- SMART CORRECTION (B/8 and Q/O) ---
def smart_correction(text):
    """
    Tries swapping B/8 and Q/O to see if a valid authorized plate exists.
    """
    if is_authorized(text):
        return text

    # Define the confusing pairs
    confusables = {
        'B': '8', '8': 'B',
        'Q': 'O', 'O': 'Q' 
    }
    
    indices = [i for i, char in enumerate(text) if char in confusables]
    
    if not indices:
        return text
        
    chars = list(text)
    options = [[c, confusables[c]] for c in [text[i] for i in indices]]
    
    # Check every possible combination
    for combo in itertools.product(*options):
        for i, idx in enumerate(indices):
            chars[idx] = combo[i]
        candidate = "".join(chars)
        
        if is_authorized(candidate):
            return candidate

    return text

# --- THREADING SETUP ---
ocr_queue = queue.Queue(maxsize=1) 
current_display_text = {} 

if not os.path.exists(log_filename):
    with open(log_filename, 'w', newline='') as f:
        csv.writer(f).writerow(['Timestamp', 'License_Plate', 'Status', 'OCR_Confidence'])

# --- HELPER FUNCTIONS ---
def preprocess_for_ocr(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    
    if width < 300:
        gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_LANCZOS4)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    h, w = thresh.shape
    border = 30 
    canvas = np.full((h + border*2, w + border*2), 255, dtype=np.uint8)
    canvas[border:h+border, border:w+border] = thresh
    return canvas

def clean_text(text):
    return re.sub(r'[^A-Z0-9]', '', text.upper())

def is_similar_to_recent(new_text, current_time):
    for past_text, last_time in seen_plates.items():
        if (current_time - last_time) < LOG_COOLDOWN:
            ratio = SequenceMatcher(None, new_text, past_text).ratio()
            if ratio >= SIMILARITY_THRESH:
                return True, past_text
    return False, None

# --- BACKGROUND WORKER THREAD ---
def ocr_worker():
    while True:
        try:
            update_authorized_list()

            data = ocr_queue.get(timeout=1) 
            img_crop, conf_yolo, box_id = data
            
            processed_plate = preprocess_for_ocr(img_crop)
            
            ocr_results = reader.readtext(processed_plate, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ123456789')
            
            for _, text, ocr_conf in ocr_results:
                raw_plate = clean_text(text)
                
                if not re.match(PLATE_REGEX, raw_plate): continue
                if ocr_conf < OCR_THRESH: continue
                
                # --- APPLY SMART CORRECTION ---
                clean_plate = smart_correction(raw_plate)
                
                current_time = time.time()
                is_duplicate, match_name = is_similar_to_recent(clean_plate, current_time)
                
                if is_authorized(clean_plate):
                    auth_status = "AUTHORIZED"
                    display_color = (0, 255, 0) 
                else:
                    auth_status = "UNAUTHORIZED"
                    display_color = (0, 0, 255) 
                
                display_text = f"{clean_plate} ({auth_status[:4]})"

                if is_duplicate:
                    # --- STABILITY CHECK ---
                    if clean_plate == match_name:
                        if match_name in plate_votes:
                            plate_votes[match_name].clear()
                    else:
                        if match_name not in plate_votes:
                            plate_votes[match_name] = {}
                        
                        plate_votes[match_name][clean_plate] = plate_votes[match_name].get(clean_plate, 0) + 1
                        
                        if plate_votes[match_name][clean_plate] >= STABILITY_VOTES_REQUIRED:
                            print(f"🔄 CORRECTION: Replacing '{match_name}' with '{clean_plate}' (Seen 10x)")
                            
                            del seen_plates[match_name]
                            seen_plates[clean_plate] = current_time
                            del plate_votes[match_name]
                            
                            match_name = clean_plate 
                            if is_authorized(clean_plate):
                                auth_status = "AUTHORIZED"
                                display_color = (0, 255, 0)
                            else:
                                auth_status = "UNAUTHORIZED"
                                display_color = (0, 0, 255)
                            
                            timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            with open(log_filename, 'a', newline='') as f:
                                csv.writer(f).writerow([
                                    timestamp_str, 
                                    clean_plate, 
                                    f"{auth_status} (CORRECTED)", 
                                    f"{ocr_conf:.2f}"
                                ])
                            print(f"📝 LOG UPDATED: {clean_plate}")

                            display_text = f"{clean_plate} ({auth_status[:4]})"
                            current_display_text[box_id] = (display_text, display_color)

                    seen_plates[match_name] = current_time
                    current_display_text[box_id] = (f"{match_name} ({auth_status[:4]})", display_color)

                else:
                    # New Plate
                    timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    with open(log_filename, 'a', newline='') as f:
                        csv.writer(f).writerow([
                            timestamp_str, 
                            clean_plate, 
                            auth_status, 
                            f"{ocr_conf:.2f}"
                        ])
                    
                    log_icon = "✅" if auth_status == "AUTHORIZED" else "⚠️"
                    print(f"{log_icon} LOGGED: {clean_plate} [{auth_status}]")
                    
                    seen_plates[clean_plate] = current_time
                    current_display_text[box_id] = (display_text, display_color)
                
                break 
            
            ocr_queue.task_done()
            
        except queue.Empty:
            continue
        except Exception as e:
            print(f"OCR Error: {e}")

# Start thread
t = threading.Thread(target=ocr_worker, daemon=True)
t.start()

# --- SOURCE SETUP ---
img_ext_list = ['.jpg','.jpeg','.png','.bmp']
vid_ext_list = ['.avi','.mov','.mp4','.mkv']

if os.path.isdir(img_source):
    source_type = 'folder'
    imgs_list = [f for f in glob.glob(img_source + '/*') if os.path.splitext(f)[1] in img_ext_list]
elif os.path.isfile(img_source):
    _, ext = os.path.splitext(img_source)
    if ext in img_ext_list: source_type = 'image'; imgs_list = [img_source]
    elif ext in vid_ext_list: source_type = 'video'; cap = cv2.VideoCapture(img_source)
elif 'usb' in img_source:
    source_type = 'usb'
    cap = cv2.VideoCapture(int(img_source[3:]))
elif 'picamera' in img_source:
    source_type = 'picamera'
    from picamera2 import Picamera2
    cap = Picamera2()
    cap.configure(cap.create_video_configuration(main={"format": 'RGB888', "size": (int(user_res.split('x')[0]), int(user_res.split('x')[1]))}))
    cap.start()

if user_res:
    resW, resH = map(int, user_res.split('x'))
    if source_type in ['video', 'usb']: cap.set(3, resW); cap.set(4, resH)
    resize = True
else: 
    resize = False

if record:
    recorder = cv2.VideoWriter('demo1.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30, (resW,resH)) if resize else None

bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106)]
img_count = 0
frame_rate_calc = 1
freq = cv2.getTickFrequency()

# --- MAIN LOOP ---
while True:
    t1 = cv2.getTickCount()

    if source_type in ['image', 'folder']:
        if img_count >= len(imgs_list): break
        frame = cv2.imread(imgs_list[img_count]); img_count += 1
    elif source_type in ['video', 'usb']:
        ret, frame = cap.read()
        if not ret: break
    elif source_type == 'picamera':
        frame = cap.capture_array()

    if resize: frame = cv2.resize(frame, (resW, resH))

    results = model(frame, verbose=False)
    detections = results[0].boxes

    for i in range(len(detections)):
        xyxy = detections[i].xyxy.cpu().numpy().squeeze().astype(int)
        xmin, ymin, xmax, ymax = xyxy
        classidx = int(detections[i].cls.item())
        classname = labels[classidx]
        conf = detections[i].conf.item()

        if conf > min_thresh:
            if 'plate' in classname.lower():
                box_id = f"{xmin}_{ymin}"
                
                if box_id in current_display_text:
                    text_label, color = current_display_text[box_id]
                else:
                    text_label = "Scanning..."
                    color = (255, 100, 0) # Orange waiting
                    
                    if not ocr_queue.full():
                        h_img, w_img, _ = frame.shape
                        pad_x = int((xmax - xmin) * 0.08)
                        pad_y = int((ymax - ymin) * 0.08)
                        crop_xmin = max(0, xmin - pad_x)
                        crop_ymin = max(0, ymin - pad_y)
                        crop_xmax = min(w_img, xmax + pad_x)
                        crop_ymax = min(h_img, ymax + pad_y)
                        
                        plate_crop = frame[crop_ymin:crop_ymax, crop_xmin:crop_xmax].copy()
                        
                        if plate_crop.size > 0:
                            ocr_queue.put((plate_crop, conf, box_id))

                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                cv2.putText(frame, text_label, (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            else:
                color = bbox_colors[classidx % 5]
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)

    if img_count % 100 == 0:
        current_display_text.clear()

    # --- FPS CALCULATION ---
    t2 = cv2.getTickCount()
    time1 = (t2 - t1) / freq
    frame_rate_calc = 1 / time1 if time1 > 0 else 0
    
    cv2.putText(frame, f'FPS: {frame_rate_calc:.2f}', (30, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('Parking Scanner', frame)
    if record and recorder: recorder.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

if source_type in ['video', 'usb']: cap.release()
elif source_type == 'picamera': cap.stop()
if record and recorder: recorder.release()
cv2.destroyAllWindows()