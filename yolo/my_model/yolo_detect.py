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
from difflib import SequenceMatcher

import cv2
import numpy as np
import easyocr
from ultralytics import YOLO

# --- CONFIGURATION ---
warnings.filterwarnings("ignore") 

# STRICTNESS SETTINGS
SIMILARITY_THRESH = 0.60  
OCR_THRESH = 0.80        
LOG_COOLDOWN = 60.0       
PLATE_REGEX = r'^[A-Z0-9]{5,8}$' 
TARGET_CLASS_NAME = 'license_plate' 

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

# --- THREADING SETUP ---
# The Queue acts as a "mailbox" between the Video Loop and the OCR Worker
ocr_queue = queue.Queue(maxsize=1) # Only hold 1 pending image to prevent lag buildup
current_display_text = {} # Shared variable to display text on screen { 'box_id': 'TEXT' }

if not os.path.exists(log_filename):
    with open(log_filename, 'w', newline='') as f:
        csv.writer(f).writerow(['Timestamp', 'License_Plate', 'OCR_Confidence', 'YOLO_Confidence'])

# --- HELPER FUNCTIONS ---
def preprocess_for_ocr(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    if width < 300:
        gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_LANCZOS4)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((2,2), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=1)
    
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
            # Wait for an image from the main loop
            data = ocr_queue.get(timeout=1) 
            img_crop, conf_yolo, box_id = data
            
            processed_plate = preprocess_for_ocr(img_crop)
            ocr_results = reader.readtext(processed_plate, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
            
            found_valid = False
            for _, text, ocr_conf in ocr_results:
                clean_plate = clean_text(text)
                
                if not re.match(PLATE_REGEX, clean_plate): continue
                if ocr_conf < OCR_THRESH: continue
                
                current_time = time.time()
                is_duplicate, match_name = is_similar_to_recent(clean_plate, current_time)
                
                if is_duplicate:
                    seen_plates[match_name] = current_time
                    # Update display to show we know this car
                    current_display_text[box_id] = (match_name, (0, 255, 255)) # Yellow
                else:
                    # New log
                    timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    with open(log_filename, 'a', newline='') as f:
                        csv.writer(f).writerow([timestamp_str, clean_plate, ocr_conf, conf_yolo])
                    print(f"✅ LOGGED: {clean_plate} (Conf: {ocr_conf:.2f})")
                    seen_plates[clean_plate] = current_time
                    # Update display to show success
                    current_display_text[box_id] = (clean_plate, (0, 255, 0)) # Green
                
                found_valid = True
                break # Only take the best read
            
            ocr_queue.task_done()
            
        except queue.Empty:
            continue
        except Exception as e:
            print(f"OCR Error: {e}")

# Start the background thread
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
                # Define a unique ID for this box detection (rough approximation)
                # We use coordinates to match the display text to the box
                box_id = f"{xmin}_{ymin}"
                
                # Check if we already have a result for this box from the thread
                if box_id in current_display_text:
                    text_label, color = current_display_text[box_id]
                else:
                    text_label = "Scanning..."
                    color = (255, 100, 0) # Blue/Orange waiting color
                    
                    # If queue is empty, send this plate to be read
                    # We check full() to ensure we don't lag the video by stacking 100 images
                    if not ocr_queue.full():
                        h_img, w_img, _ = frame.shape
                        pad_x = int((xmax - xmin) * 0.08)
                        pad_y = int((ymax - ymin) * 0.08)
                        crop_xmin = max(0, xmin - pad_x)
                        crop_ymin = max(0, ymin - pad_y)
                        crop_xmax = min(w_img, xmax + pad_x)
                        crop_ymax = min(h_img, ymax + pad_y)
                        
                        plate_crop = frame[crop_ymin:crop_ymax, crop_xmin:crop_xmax].copy() # .copy() is crucial for threads
                        
                        if plate_crop.size > 0:
                            # Send to background thread!
                            ocr_queue.put((plate_crop, conf, box_id))

                # Draw Box & Text
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                cv2.putText(frame, text_label, (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            else:
                # Other objects (not plates)
                color = bbox_colors[classidx % 5]
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)

    # Clean up old IDs from display dict to save RAM (Simple cleanup every 100 frames)
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