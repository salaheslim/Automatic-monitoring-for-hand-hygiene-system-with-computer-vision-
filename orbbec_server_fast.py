import sys
sys.path.insert(0, '/home/salah/pyorbbecsdk/install/lib')
import pyorbbecsdk as ob
import cv2
import numpy as np
import socket
import struct
import time

ctx = ob.Context()
device_list = ctx.query_devices()
device = device_list[0]
pipeline = ob.Pipeline(device)
config = ob.Config()

profile_list = pipeline.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR)
color_profile = profile_list.get_default_video_stream_profile()
config.enable_stream(color_profile)

depth_profile_list = pipeline.get_stream_profile_list(ob.OBSensorType.DEPTH_SENSOR)
depth_profile = depth_profile_list.get_default_video_stream_profile()
config.enable_stream(depth_profile)

pipeline.start(config)
print("Fast Orbbec server started")

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind(('localhost', 9999))
server.listen(1)
print("Waiting for client...")
conn, addr = server.accept()
print(f"Client connected: {addr}")

frame_count = 0
start = time.time()

while True:
    frames = pipeline.wait_for_frames(100)
    if frames is None:
        continue

    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()
    if color_frame is None:
        continue

    # Get colour image
    color_data = np.frombuffer(color_frame.get_data(), dtype=np.uint8)
    w = color_frame.get_width()
    h = color_frame.get_height()
    color_image = cv2.imdecode(color_data, cv2.IMREAD_COLOR)
    if color_image is None:
        try:
            color_image = color_data.reshape((h, w, 3))
        except:
            continue

    color_image = cv2.resize(color_image, (640, 480))

    # Compress colour as JPEG
    _, jpeg = cv2.imencode('.jpg', color_image, 
                           [cv2.IMWRITE_JPEG_QUALITY, 80])
    jpeg_bytes = jpeg.tobytes()

    # Get depth as compressed uint16
    depth_bytes = b''
    if depth_frame is not None:
        depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
        dw = depth_frame.get_width()
        dh = depth_frame.get_height()
        try:
            depth_map = depth_data.reshape((dh, dw))
            depth_small = cv2.resize(depth_map.astype(np.float32),
                                    (320, 240)).astype(np.uint16)
            depth_bytes = depth_small.tobytes()
        except:
            pass

    # Send: [jpeg_size(4)][jpeg][depth_size(4)][depth]
    try:
        header = struct.pack('II', len(jpeg_bytes), len(depth_bytes))
        conn.sendall(header + jpeg_bytes + depth_bytes)
    except:
        break

    frame_count += 1
    fps = frame_count / (time.time() - start)
    if frame_count % 30 == 0:
        print(f"Server FPS: {fps:.1f}")

pipeline.stop()
conn.close()
server.close()
