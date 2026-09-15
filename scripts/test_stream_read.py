import av
import requests
import io
import cv2

print("=== TESTING PYAV STREAM DECODER ===")
url = "https://live.corp8.cloud/stream/1"

try:
    container = av.open(url, mode="r", timeout=5.0)
    print("PyAV Container opened successfully:", container)
    for frame in container.decode(video=0):
        img = frame.to_ndarray(format="bgr24")
        print(f"Decoded live frame successfully! Shape: {img.shape}")
        cv2.imwrite("output/cam1_live_frame.jpg", img)
        break
    container.close()
except Exception as e:
    print(f"PyAV Error: {e}")
