import urllib.request
import cv2
import numpy as np
import time

url = "http://127.0.0.1:8000/api/v1/stream/frame/CAM_09"
print("Monitoring CAM_09 stream for vehicles with plates...")

captured = 0
start_t = time.time()
while time.time() - start_t < 25 and captured < 5:
    try:
        resp = urllib.request.urlopen(url, timeout=2)
        arr = np.asarray(bytearray(resp.read()), dtype=np.uint8)
        frame = cv2.imdecode(arr, -1)
        if frame is None:
            time.sleep(0.1)
            continue

        # Look for green / gold overlay or text indicating vehicle
        # Let's save frames that have vehicles in them
        # Vehicle presence can be tested by frame variance or motion or color
        # In fact, the live pipeline overlays yellow/gold boxes on vehicles with plates!
        # Let's check for yellow/gold overlay pixels (BGR: [0, 215, 255] or [0, 255, 255])
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Yellow mask
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([35, 255, 255])
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        yellow_px = cv2.countNonZero(mask)

        # Also check for green boxes
        lower_green = np.array([40, 100, 100])
        upper_green = np.array([80, 255, 255])
        mask_g = cv2.inRange(hsv, lower_green, upper_green)
        green_px = cv2.countNonZero(mask_g)

        if yellow_px > 300 or green_px > 300:
            captured += 1
            fn = f"cam09_vehicle_event_{captured}.jpg"
            cv2.imwrite(fn, frame)
            print(f"Captured vehicle event {captured}: Yellow={yellow_px}px, Green={green_px}px -> saved {fn}")
            time.sleep(1.0)
    except Exception as e:
        time.sleep(0.2)

print(f"Done. Captured {captured} vehicle events.")
