import os
import sys
import csv
import glob
import time
import argparse
import threading
from datetime import datetime

import cv2
import numpy as np
from ultralytics import YOLO

# Define and parse user input arguments

parser = argparse.ArgumentParser(description='Real-time weapon detection with alerts and evidence logging')
parser.add_argument('--model', help='Path to YOLO model file', default='best.pt')
parser.add_argument('--source', help='Image file, image folder, video file, USB camera ("usb0") or Picamera ("picamera0")',
                    default='usb0')
parser.add_argument('--thresh', help='Minimum confidence threshold (example: "0.4")', type=float, default=0.5)
parser.add_argument('--resolution', help='Resolution in WxH to display results at (example: "640x480")', default=None)
parser.add_argument('--record', help='Record results from video or webcam as "demo1.avi". Requires --resolution.',
                    action='store_true')
parser.add_argument('--confirm-frames', help='Consecutive frames with a weapon before the alert fires', type=int, default=5)
parser.add_argument('--cooldown', help='Seconds between saved evidence snapshots', type=float, default=3.0)
parser.add_argument('--out-dir', help='Folder for evidence snapshots and the CSV log', default='detections')
parser.add_argument('--no-beep', help='Disable the alert sound', action='store_true')
parser.add_argument('--no-track', help='Use plain detection instead of tracking', action='store_true')

args = parser.parse_args()

model_path = args.model
img_source = args.source
min_thresh = args.thresh
user_res = args.resolution
record = args.record

# Check if model file exists and is valid
if not os.path.exists(model_path):
    print('ERROR: Model path is invalid or model was not found. Make sure the model filename was entered correctly.')
    sys.exit(0)

# Load the model into memory and get labelmap
model = YOLO(model_path, task='detect')
labels = model.names

# Per-class box colours (BGR)
class_colors = {
    'gun': (0, 0, 255),             # red
    'knife': (0, 165, 255),         # orange
    'heavy-weapon': (128, 0, 128),  # purple
}
default_color = (200, 200, 200)

# Parse input to determine if image source is a file, folder, video, or USB camera
img_ext_list = ['.jpg', '.jpeg', '.png', '.bmp']
vid_ext_list = ['.avi', '.mov', '.mp4', '.mkv', '.wmv']

if os.path.isdir(img_source):
    source_type = 'folder'
elif os.path.isfile(img_source):
    _, ext = os.path.splitext(img_source)
    if ext.lower() in img_ext_list:
        source_type = 'image'
    elif ext.lower() in vid_ext_list:
        source_type = 'video'
    else:
        print(f'File extension {ext} is not supported.')
        sys.exit(0)
elif img_source.startswith('usb'):
    source_type = 'usb'
    usb_idx = int(img_source[3:])
elif img_source.startswith('picamera'):
    source_type = 'picamera'
    picam_idx = int(img_source[8:])
else:
    print(f'Input {img_source} is invalid. Please try again.')
    sys.exit(0)

is_stream = source_type in ['video', 'usb', 'picamera']
use_tracking = is_stream and not args.no_track

# Single images have no "consecutive frames", so alert immediately
confirm_frames = max(1, args.confirm_frames) if is_stream else 1

# Parse user-specified display resolution
resize = False
if user_res:
    resize = True
    resW, resH = int(user_res.split('x')[0]), int(user_res.split('x')[1])

# Check if recording is valid and set up recording
if record:
    if source_type not in ['video', 'usb']:
        print('Recording only works for video and camera sources. Please try again.')
        sys.exit(0)
    if not user_res:
        print('Please specify resolution to record video at.')
        sys.exit(0)

    record_name = 'demo1.avi'
    record_fps = 30
    recorder = cv2.VideoWriter(record_name, cv2.VideoWriter_fourcc(*'MJPG'), record_fps, (resW, resH))

# Set up evidence folder and CSV log
os.makedirs(args.out_dir, exist_ok=True)
log_path = os.path.join(args.out_dir, 'detections_log.csv')
new_log = not os.path.exists(log_path)
log_file = open(log_path, 'a', newline='')
log_writer = csv.writer(log_file)
if new_log:
    log_writer.writerow(['timestamp', 'class', 'confidence', 'track_id', 'snapshot'])


def beep():
    """Play a short alert sound without blocking the video loop."""
    def _play():
        try:
            import winsound
            winsound.Beep(1000, 400)
        except Exception:
            print('\a', end='', flush=True)
    threading.Thread(target=_play, daemon=True).start()


# Load or initialize image source
if source_type == 'image':
    imgs_list = [img_source]
elif source_type == 'folder':
    imgs_list = []
    for file in glob.glob(img_source + '/*'):
        _, file_ext = os.path.splitext(file)
        if file_ext.lower() in img_ext_list:
            imgs_list.append(file)
elif source_type == 'video' or source_type == 'usb':
    cap_arg = img_source if source_type == 'video' else usb_idx
    cap = cv2.VideoCapture(cap_arg)

    # Set camera or video resolution if specified by user
    if user_res:
        cap.set(3, resW)
        cap.set(4, resH)

elif source_type == 'picamera':
    from picamera2 import Picamera2
    cap = Picamera2()
    cap.configure(cap.create_video_configuration(main={"format": 'RGB888', "size": (resW, resH)}))
    cap.start()

# Initialize control and status variables
avg_frame_rate = 0
frame_rate_buffer = []
fps_avg_len = 200
img_count = 0
weapon_streak = 0          # consecutive frames containing a weapon
alert_active = False
last_snapshot_time = 0.0

# Begin inference loop
while True:

    t_start = time.perf_counter()

    # Load frame from image source
    if source_type == 'image' or source_type == 'folder':
        if img_count >= len(imgs_list):
            print('All images have been processed. Exiting program.')
            break
        frame = cv2.imread(imgs_list[img_count])
        img_count = img_count + 1

    elif source_type == 'video':
        ret, frame = cap.read()
        if not ret:
            print('Reached end of the video file. Exiting program.')
            break

    elif source_type == 'usb':
        ret, frame = cap.read()
        if (frame is None) or (not ret):
            print('Unable to read frames from the camera. This indicates the camera is disconnected or not working. Exiting program.')
            break

    elif source_type == 'picamera':
        frame = cap.capture_array()
        if frame is None:
            print('Unable to read frames from the Picamera. This indicates the camera is disconnected or not working. Exiting program.')
            break

    # Resize frame to desired display resolution
    if resize:
        frame = cv2.resize(frame, (resW, resH))

    # Run inference; YOLO drops boxes below the threshold itself
    if use_tracking:
        results = model.track(frame, conf=min_thresh, persist=True, tracker='bytetrack.yaml', verbose=False)
    else:
        results = model(frame, conf=min_thresh, verbose=False)

    detections = results[0].boxes
    track_ids = detections.id.int().tolist() if detections.id is not None else [None] * len(detections)

    frame_detections = []  # (classname, conf, track_id) for this frame

    # Go through each detection and draw it
    for i in range(len(detections)):

        xmin, ymin, xmax, ymax = detections[i].xyxy.cpu().numpy().squeeze().astype(int)
        classidx = int(detections[i].cls.item())
        classname = labels[classidx]
        conf = detections[i].conf.item()
        track_id = track_ids[i]

        frame_detections.append((classname, conf, track_id))

        color = class_colors.get(classname, default_color)
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)

        label = f'{classname}: {int(conf*100)}%'
        if track_id is not None:
            label = f'#{track_id} {label}'
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_ymin = max(ymin, labelSize[1] + 10)  # Make sure not to draw label too close to top of window
        cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), color, cv2.FILLED)
        cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    object_count = len(frame_detections)

    # Only raise the alert after a weapon has been seen in several consecutive frames
    weapon_streak = weapon_streak + 1 if object_count > 0 else 0
    was_active = alert_active
    alert_active = weapon_streak >= confirm_frames

    if alert_active and not was_active and not args.no_beep:
        beep()

    # Draw alert banner
    if alert_active:
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, h - 50), (w, h), (0, 0, 255), cv2.FILLED)
        text = 'WEAPON DETECTED'
        textSize, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        cv2.putText(frame, text, ((w - textSize[0]) // 2, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    # Draw framerate and object count
    if is_stream:
        cv2.putText(frame, f'FPS: {avg_frame_rate:0.2f}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 255), 2)
    cv2.putText(frame, f'Weapons in frame: {object_count}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 255), 2)

    # Save evidence snapshot and log it (rate-limited by --cooldown)
    now = time.time()
    if alert_active and (now - last_snapshot_time >= args.cooldown or not is_stream):
        last_snapshot_time = now
        stamp = datetime.now()
        snap_name = f'weapon_{stamp.strftime("%Y%m%d_%H%M%S_%f")[:-3]}.jpg'
        cv2.imwrite(os.path.join(args.out_dir, snap_name), frame)
        for classname, conf, track_id in frame_detections:
            log_writer.writerow([stamp.isoformat(timespec='milliseconds'), classname, f'{conf:.3f}',
                                 '' if track_id is None else track_id, snap_name])
        log_file.flush()
        print(f'[{stamp.strftime("%H:%M:%S")}] Weapon detected: ' +
              ', '.join(f'{c} ({p:.0%})' for c, p, _ in frame_detections) + f' -> saved {snap_name}')

    cv2.imshow('Weapon detection', frame)
    if record:
        recorder.write(frame)

    # If inferencing on individual images, wait for user keypress before moving to next image
    if is_stream:
        key = cv2.waitKey(5)
    else:
        key = cv2.waitKey()

    if key == ord('q') or key == ord('Q'):  # Press 'q' to quit
        break
    elif key == ord('s') or key == ord('S'):  # Press 's' to pause inference
        cv2.waitKey()
    elif key == ord('p') or key == ord('P'):  # Press 'p' to save a picture of results on this frame
        cv2.imwrite('capture.png', frame)

    # Calculate average FPS over recent frames
    t_stop = time.perf_counter()
    frame_rate_buffer.append(float(1 / (t_stop - t_start)))
    if len(frame_rate_buffer) > fps_avg_len:
        frame_rate_buffer.pop(0)
    avg_frame_rate = np.mean(frame_rate_buffer)


# Clean up
print(f'Average pipeline FPS: {avg_frame_rate:.2f}')
print(f'Evidence log: {log_path}')
if source_type == 'video' or source_type == 'usb':
    cap.release()
elif source_type == 'picamera':
    cap.stop()
if record:
    recorder.release()
log_file.close()
cv2.destroyAllWindows()
