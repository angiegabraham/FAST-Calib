''''
Uses ROS1 Bag

Usage: python3 scripts/overlay_test.py --bag /path/to/bag --out_dir /path/to/output_directory
'''

import numpy as np
import struct
import cv2
import argparse
import os
from rosbags.rosbag1 import Reader
from rosbags.typesys import get_typestore, Stores

def overlay(bag_path, output_path):
    typestore = get_typestore(Stores.ROS1_NOETIC)

    # extrinsics from calibration (trial 2)
    R = np.array([[-0.994140,  0.081789, -0.070692],
                [ 0.064672, -0.074052, -0.995155],
                [-0.086628, -0.993895,  0.068329]])
    t = np.array([0.054925, 0.254230, 2.868036])

    # camera intrinsics
    fx, fy = 605.024536, 605.381104
    cx, cy = 323.839386, 246.221069
    K    = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    dist = np.zeros(5)

    img_topic   = '/camera_1/camera_1/color/image_raw'
    lidar_topic = '/registered_scan'
    MAX_LIDAR_FRAMES = 20

    image = None
    lidar_pts = []
    lidar_frame_count = 0
    got_image = False

    with Reader(bag_path) as reader:
        for conn, ts, rawdata in reader.messages():
            if conn.topic == img_topic and not got_image:
                msg = typestore.deserialize_ros1(rawdata, conn.msgtype)
                h, w = int(msg.height), int(msg.width)
                raw = bytes(msg.data)
                enc = msg.encoding
                if enc == 'rgb8':
                    arr = np.frombuffer(raw, np.uint8).reshape(h, w, 3)
                    image = arr[:, :, ::-1].copy()
                elif enc == 'bgr8':
                    image = np.frombuffer(raw, np.uint8).reshape(h, w, 3).copy()
                got_image = True
                print(f"Got image: {w}x{h} {enc}")

            if conn.topic == lidar_topic and lidar_frame_count < MAX_LIDAR_FRAMES:
                msg = typestore.deserialize_ros1(rawdata, conn.msgtype)
                point_step = msg.point_step
                data = bytes(msg.data)
                n = len(data) // point_step
                for i in range(n):
                    x, y, z = struct.unpack_from('fff', data, i * point_step)
                    if np.isfinite(x) and np.isfinite(y) and np.isfinite(z):
                        lidar_pts.append([x, y, z])
                lidar_frame_count += 1

            if got_image and lidar_frame_count >= MAX_LIDAR_FRAMES:
                break

    print(f"LiDAR frames: {lidar_frame_count}, total points: {len(lidar_pts)}")

    if image is None:
        print("ERROR: no image found")
        exit()

    # transform to camera frame
    pts = np.array(lidar_pts)
    pts_cam = (R @ pts.T).T + t

    # filter in front of camera and not beyond 2m
    mask = (pts_cam[:, 2] > 0.1) & (pts_cam[:, 2] < 2.0)

    pts_cam = pts_cam[mask]
    print(f"Points in front of camera: {len(pts_cam)}")

    # project onto image 
    pts_proj, _ = cv2.projectPoints(
        pts_cam.astype(np.float32),
        np.zeros(3), np.zeros(3), K, dist)
    uvs = pts_proj.reshape(-1, 2)

    h, w = image.shape[:2]
    overlay = image.copy()

    depths = pts_cam[:, 2]
    d_min = 0.5
    d_max = 3.0

    count = 0
    for (u, v), d in zip(uvs, depths):
        ui, vi = int(u), int(v)
        if 0 <= ui < w and 0 <= vi < h:
            norm = float(np.clip((d - d_min) / (d_max - d_min + 1e-6), 0, 1))
            color = (
                int(255 * (1 - norm)),
                int(255 * (1 - abs(2 * norm - 1))),
                int(255 * norm)
            )
            cv2.circle(overlay, (ui, vi), 1, color, -1)
            count += 1

    print(f"Points projected onto image: {count}")
    cv2.imwrite(output_path, overlay)
    cv2.imshow('LiDAR overlay. Press any key to quit', overlay)
    while True:
        key = cv2.waitKey(100)
        if key != -1:
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--bag', type=str, required=True, help='path to bag')
    parser.add_argument('--out_dir', type=str, required=True, help='path to output dir')

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    bag_name = os.path.splitext(os.path.basename(args.bag))[0]
    output_path = os.path.join(args.out_dir, f'{bag_name}_overlay.jpg')

    overlay(args.bag, output_path)