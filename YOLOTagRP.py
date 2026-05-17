import os

if 'OPENBLAS_CORETYPE' in os.environ:
    del os.environ['OPENBLAS_CORETYPE']

import cv2
import numpy as np
from pupil_apriltags import Detector
from ultralytics import YOLO

# --- CONFIGURATION ---
os.environ['OPENBLAS_CORETYPE'] = 'ARMV8'
TAG_SIZE = 0.18  # The physical width of your tag in METERS (e.g., 0.15 = 15cm)
WIDTH = 1280
HEIGHT = 720
CAMERA_INDEX = 1  # 0 is usually the first USB cam; 1 if you have a built-in one
CONF_THRESHOLD = 0.75
# [focal_length_x, 0, center_x, 0, focal_length_y, center_y, 0, 0, 1]
# For better accuracy, you should calibrate your specific camera.
params = [1386, 1384, WIDTH // 2, HEIGHT // 2]
at_detector = Detector(families="tag36h11")
cap = cv2.VideoCapture(CAMERA_INDEX) # Change to 1 for external camera
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    print(f"Error: Could not open USB camera at index {CAMERA_INDEX}")
    exit()

person_count = 0

try:
    # Force CPU to rule out GPU/Vulkan driver crashes
    model = YOLO("yolo11n.pt", task='detect')
    model.to('cpu')
    print("Step C: Model loaded successfully!")
except Exception as e:
    print(f"Failed at Step B: {e}")

print("USB Camera Stream Started!")

while True:
    ret, frame = cap.read()
    
    if not ret: break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Crucial: estimate_tag_pose must be True
    april_results = at_detector.detect(gray, 
                                 estimate_tag_pose=True, 
                                 camera_params=params, 
                                 tag_size=TAG_SIZE)
    
                                 
    YOLO_results = model(frame, imgsz=320, conf=CONF_THRESHOLD, verbose=False)

    for r in april_results:  
        def get_real_coordinates(r):
            def calculate_homography(src_pts, dst_pts):
                """
                src_pts: 4 points on the ideal tag [(x1,y1), (x2,y2)...]
                dst_pts: 4 corner pixels from the camera [(u1,v1), (u2,v2)...]
                """
                A = []
                for i in range(4):
                    x, y = src_pts[i]
                    u, v = dst_pts[i]
                    A.append([-x, -y, -1, 0, 0, 0, x*u, y*u, u])
                    A.append([0, 0, 0, -x, -y, -1, x*v, y*v, v])
                A = np.array(A)
                    
                # Solve Ah = 0 using SVD
                U, S, Vt = np.linalg.svd(A)
                    
                # The solution is the last row of Vt (which is the last column of V)
                h = Vt[-1, :]
                    
                # Reshape to 3x3 and normalize so h[2,2] == 1
                H = h.reshape((3, 3))
                return H / H[2, 2]
            corners = r.corners
            tag_coordinates = [TAG_SIZE / 2, TAG_SIZE / 2], [-TAG_SIZE / 2, TAG_SIZE / 2], [-TAG_SIZE / 2, -TAG_SIZE / 2], [TAG_SIZE / 2, -TAG_SIZE / 2]
            H = calculate_homography(tag_coordinates, corners)
             
            
            

        rel_x, rel_y, rel_z = r.pose_t.flatten()

        # 1. Print to Terminal
        print(f"Tag ID {r.tag_id} -> X: {rel_x:.2f}m, Y: {rel_y:.2f}m, Z: {rel_z:.2f}m")

        # 2. Draw on Screen
        cX, cY = int(r.center[0]), int(r.center[1])
        
        # Display the coordinates next to the tag
        pos_text = f"X:{rel_x:.2f} Y:{rel_y:.2f} Z:{rel_z:.2f}"
        cv2.putText(frame, pos_text, (cX - 80, cY + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        # Draw the bounding box for visual clarity
        pts = r.corners.reshape((-1, 1, 2)).astype(np.int32)
        cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
        
    
        
    detected_this_frame = False
    # Instead of results[1], iterate through the list of result objects
    for result in YOLO_results:
        # Use 'any' for a cleaner check
        if any(model.names[int(box.cls[0])] == 'person' for box in result.boxes):
            person_count += 1
        else:
            person_count = 0
        
    if person_count >= 3:
        print("CONFIRMED: Person detected.")
        # Reset or add a trigger here (e.g., take a high-res photo)
        person_count = 0

    cv2.imshow("Relative Positioning", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
print("Releasing camera and closing...")
cap.release()
cv2.destroyAllWindows()