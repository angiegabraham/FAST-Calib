#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
功能：
1) 自动检测 rosbag 中雷达点云类型：
   - sensor_msgs/PointCloud2  (如 /hesai/pandar)
   - livox_ros_driver/CustomMsg (如 /livox/lidar)
2) 按各自的解析方式把点云导出成一个带 intensity 的 PCD 文件 (x y z intensity, ASCII)
3) 使用 Open3D 对该 PCD 进行交互选点（至少 4 个点），并根据 4 个点计算包围范围，
   保存为同名 .txt 文件。

依赖：
    - rosbag
    - sensor_msgs.point_cloud2
    - open3d, numpy

用法示例：
    python FAST-Calib-tool.py
    python FAST-Calib-tool.py /path/to/data.bag /path/to/output_dir
"""

import os
import sys
import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
from pathlib import Path
# import sensor_msgs_py.point_cloud2 as pc2
# from sensor_msgs.msg import PointCloud2
# from std_msgs.msg import Header
# from rclpy.time import Time
# from sensor_msgs.msg import PointField
import struct 
import open3d as o3d
import open3d.core as o3c
import argparse

# ===================== 通用：保存 PCD =====================

def save_pcd_with_intensity(points, intensities, output_path):
    """
    保存点云为带 intensity 字段的 PCD 文件 (ASCII 格式)
    points: list/ndarray of [x, y, z]
    intensities: list/ndarray of intensity
    """
    N = len(points)
    header = f"""# .PCD v0.7 - Point Cloud Data file format
VERSION 0.7
FIELDS x y z intensity
SIZE 4 4 4 4
TYPE F F F F
COUNT 1 1 1 1
WIDTH {N}
HEIGHT 1
POINTS {N}
DATA ascii
"""
    with open(output_path, 'w') as f:
        f.write(header)
        for (x, y, z), inten in zip(points, intensities):
            f.write(f"{x} {y} {z} {inten}\n")
    print(f"[PCD] 保存带 intensity 字段的点云到: {output_path}")

# ===================== 情况 1：PointCloud2 =====================

def find_intensity_field(msg):
    candidates = ["intensity", "reflectivity", "i", "ref"]
    for field in msg.fields:
        if field.name.lower() in candidates:
            return field.name
    return None

# def convert_rosbags_msg_to_native(msg_ds):
#     msg = PointCloud2()
#     # Convert header
#     msg.header = Header()
#     msg.header.frame_id = msg_ds.header.frame_id
#     msg.header.stamp = Time(seconds=msg_ds.header.stamp.sec, 
#         nanoseconds=msg_ds.header.stamp.nanosec).to_msg()
#     msg.height = msg_ds.height
#     msg.width = msg_ds.width
#     # Convert fields
#     msg.fields = []
#     for field in msg_ds.fields:
#         msg.fields.append(PointField())
#         msg.fields[-1].name = field.name
#         msg.fields[-1].offset = field.offset
#         msg.fields[-1].datatype = field.datatype
#         msg.fields[-1].count = field.count

#     msg.is_bigendian = msg_ds.is_bigendian
#     msg.point_step = msg_ds.point_step
#     msg.row_step = msg_ds.row_step
#     msg.data = msg_ds.data.tolist()
#     msg.is_dense = msg_ds.is_dense
#     return msg

# def convert_pointcloud2_bag_to_pcd(
#     bag_file,
#     output_dir,
#     topic_name="/sensor_scan",                        # 如有不同，可改成 topic 名称
#     pcd_name="sensor_PointCloud2_inten_ascii.pcd"
# ):
#     """
#     将 rosbag 中 PointCloud2 类型的点云合并导出为一个 PCD 文件。
#     保持原始雷达坐标，不做坐标变换。
#     """
#     print(f"[Bag] 打开 rosbag: {bag_file}")
#     bagpath = Path(bag_file)
#     typestore = get_typestore(Stores.ROS2_HUMBLE)

#     # 1) 先检测强度字段
#     intensity_field = None
#     with AnyReader([bagpath], default_typestore=typestore) as reader:
#         for conn, ts, rawdata in reader.messages():
#             if conn.msgtype == "sensor_msgs/msg/PointCloud2":
#                 msg = reader.deserialize(rawdata, conn.msgtype)
#                 intensity_field = find_intensity_field(msg)
#                 if intensity_field:
#                     print(f"[Bag] 检测到 intensity 字段: {intensity_field}")
#                 break

#     if not intensity_field:
#         print("[ERROR] 未找到强度字段! 退出 PointCloud2 转换。", file=sys.stderr)
#         bag.close()
#         return None

#     # 2) 读取指定 topic 的所有点云
#     all_points = []
#     all_intensities = []

#     print(f"[Bag] 开始从 topic '{topic_name}' 读取 PointCloud2 点云...")
#     # count = 0
#     with AnyReader([bagpath], default_typestore=typestore) as reader:
#         connections = [x for x in reader.connections if x.topic == topic_name]
#         for conn, ts, rawdata in reader.messages(connections=connections):
#             if conn.msgtype == "sensor_msgs/msg/PointCloud2":
#                 try:
#                     field_names = ("x", "y", "z", intensity_field)
#                     msg_ds = reader.deserialize(rawdata, conn.msgtype)
#                     msg = convert_rosbags_msg_to_native(msg_ds)
#                     for point in pc2.read_points(msg, field_names=field_names, skip_nans=True):
#                         all_points.append([point[0], point[1], point[2]])
#                         all_intensities.append(point[3])  # 强度是第四个字段
#                     # count += 1
#                     # if count == 5:
#                     #     break
#                 except Exception as e:
#                     print(f"[ERROR] 读取错误: {str(e)}", file=sys.stderr)
#                     continue

#     if not all_points:
#         print("[ERROR] 未找到 PointCloud2 点云数据！", file=sys.stderr)
#         return None

#     output_path = os.path.join(output_dir, pcd_name)
#     save_pcd_with_intensity(all_points, all_intensities, output_path)
#     return output_path

def read_points_numpy(msg_ds, field_names, has_intensity=True):
    field_map = {}
    dtype_map = {
        1: np.int8, 2: np.uint8, 3: np.int16, 4: np.uint16,
        5: np.int32, 6: np.uint32, 7: np.float32, 8: np.float64
    }
    for field in msg_ds.fields:
        if field.name in field_names:
            field_map[field.name] = (field.offset, dtype_map[field.datatype])
    point_step = msg_ds.point_step
    data = np.frombuffer(msg_ds.data.tobytes(), dtype=np.uint8)
    n_points = msg_ds.width * msg_ds.height
    results = []
    for i in range(n_points):
        row = data[i * point_step:(i + 1) * point_step]
        point = []
        for name in field_names:
            offset, dt = field_map[name]
            val = np.frombuffer(row[offset:offset + np.dtype(dt).itemsize], dtype=dt)[0]
            point.append(float(val))
        if any(np.isnan(v) for v in point[:3]):
            continue
        results.append(point)
    return results

def convert_pointcloud2_bag_to_pcd(
    bag_file,
    output_dir,
    topic_name="/sensor_scan",
    pcd_name="sensor_PointCloud2_inten_ascii.pcd"
):
    print(f"[Bag] 打开 rosbag: {bag_file}")
    bagpath = Path(bag_file)
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    intensity_field = None
    with AnyReader([bagpath], default_typestore=typestore) as reader:
        for conn, ts, rawdata in reader.messages():
            if conn.msgtype == "sensor_msgs/msg/PointCloud2":
                msg = reader.deserialize(rawdata, conn.msgtype)
                intensity_field = find_intensity_field(msg)
                if intensity_field:
                    print(f"[Bag] 检测到 intensity 字段: {intensity_field}")
                break
    if not intensity_field:
        print("[WARN] No intensity field, using default 1.0!")
    has_intensity = intensity_field is not None
    field_names = ("x", "y", "z") if not has_intensity else ("x", "y", "z", intensity_field)
    all_points = []
    all_intensities = []
    print(f"[Bag] 开始从 topic '{topic_name}' 读取 PointCloud2 点云...")
    with AnyReader([bagpath], default_typestore=typestore) as reader:
        connections = [x for x in reader.connections if x.topic == topic_name]
        for conn, ts, rawdata in reader.messages(connections=connections):
            if conn.msgtype == "sensor_msgs/msg/PointCloud2":
                try:
                    msg_ds = reader.deserialize(rawdata, conn.msgtype)
                    points = read_points_numpy(msg_ds, field_names, has_intensity=has_intensity)
                    for p in points:
                        all_points.append([p[0], p[1], p[2]])
                        all_intensities.append(p[3] if has_intensity else 1.0)
                except Exception as e:
                    print(f"[ERROR] 读取错误: {str(e)}", file=sys.stderr)
                    continue
    if not all_points:
        print("[ERROR] 未找到 PointCloud2 点云数据！", file=sys.stderr)
        return None
    output_path = os.path.join(output_dir, pcd_name)
    save_pcd_with_intensity(all_points, all_intensities, output_path)
    return output_path

# ===================== 情况 2：Livox CustomMsg =====================

# def parse_livox_custom_msg(msg):
#     """
#     从 livox_ros_driver/CustomMsg 中解析 x, y, z, reflectivity
#     假设 msg.points 是 CustomPoint 对象列表，字段为 x, y, z, reflectivity
#     """
#     points = []
#     intensities = []

#     for pt in msg.points:
#         points.append([pt.x, pt.y, pt.z])
#         intensities.append(pt.reflectivity)

#     return points, intensities

# def convert_livox_custom_bag_to_pcd(
#     bag_file,
#     output_dir,
#     topic_name="/livox/lidar",                     # 如有不同，可改成 topic 名称
#     pcd_name="livox_CustomMsg_inten_ascii.pcd"
# ):
#     """
#     将 rosbag 中 livox_ros_driver/CustomMsg 类型的点云合并导出为一个 PCD 文件。
#     保持原始雷达坐标，不做坐标变换。
#     """
#     print(f"[Bag] 打开 rosbag: {bag_file}")
#     bag = rosbag.Bag(bag_file, "r")

#     all_points = []
#     all_intensities = []

#     print(f"[Bag] 开始从 topic '{topic_name}' 读取 CustomMsg 点云...")

#     for topic, msg, t in bag.read_messages(topics=[topic_name]):
#         if msg._type == "livox_ros_driver/CustomMsg":
#             pts, intens = parse_livox_custom_msg(msg)
#             all_points.extend(pts)
#             all_intensities.extend(intens)

#     bag.close()

#     if not all_points:
#         print("[ERROR] 未找到 Livox CustomMsg 点云数据!", file=sys.stderr)
#         return None

#     output_path = os.path.join(output_dir, pcd_name)
#     intensities = np.array(all_intensities, dtype=np.float32)
#     save_pcd_with_intensity(all_points, intensities, output_path)
#     return output_path

# ===================== 自动检测：这个 bag 用哪种方式 =====================

def detect_lidar_msg_type(bag_file):
    """
    在 bag 里扫一圈，检测是否有 PointCloud2 或 Livox CustomMsg。
    返回：
        "PointCloud2", "CustomMsg", 或 None
    如果两种都有，默认优先 PointCloud2,并打印提示。
    """
    has_pc2 = False
    has_livox = False

    print(f"[Detect] 扫描 bag: {bag_file}")
    bagpath = Path(bag_file)
    typestore = get_typestore(Stores.ROS2_HUMBLE)

    with AnyReader([bagpath], default_typestore=typestore) as reader:
        for conn, ts, rawdata in reader.messages():
            if conn.msgtype == "sensor_msgs/msg/PointCloud2":
                has_pc2 = True
            elif conn.msgtype == "livox_ros_driver/msg/CustomMsg":
                has_livox = True

            if has_pc2 and has_livox:
                break

    if has_pc2 and has_livox:
        print("[Detect] 同时检测到 PointCloud2 和 Livox CustomMsg, 默认使用 PointCloud2。")
        return "PointCloud2"
    elif has_pc2:
        print("[Detect] 检测到 PointCloud2 点云。")
        return "PointCloud2"
    elif has_livox:
        print("[Detect] 检测到 Livox CustomMsg 点云。")
        return "CustomMsg"
    else:
        print("[Detect] 未检测到 PointCloud2 或 Livox CustomMsg 点云。")
        return None

# ===================== Open3D 交互选点 & 保存范围 =====================

def select_and_save_points(pcd_folder, target_pcd_name):
    """
    在给定目录中读取指定 PCD 文件，用 Open3D 交互式选点并保存范围。
    """
    pcd_path = os.path.join(pcd_folder, target_pcd_name)
    if not os.path.isfile(pcd_path):
        print(f"[ERROR] 指定的 PCD 文件不存在: {pcd_path}", file=sys.stderr)
        return

    # 读取点云
    pcd = o3d.io.read_point_cloud(pcd_path)
    if not pcd.has_points():
        print(f"[ERROR] {target_pcd_name} 中没有点云数据，已跳过", file=sys.stderr)
        return

    print(f"\n正在处理: {target_pcd_name}")
    print("请在可视化窗口中按住 Shift 用鼠标左键选择点(至少4个)，然后按 Q 键关闭窗口")

    # 创建可视化窗口并添加点云
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name=f"选择点 - {target_pcd_name}")
    vis.add_geometry(pcd)

    # 等待用户交互（Shift+左键选点, Q 退出）
    vis.run()
    vis.destroy_window()

    # 获取用户选择的点的索引
    selected_indices = vis.get_picked_points()

    if not selected_indices:
        print(f"[ERROR] 未选择任何点，{target_pcd_name} 没有保存文件", file=sys.stderr)
        return

    if len(selected_indices) < 4:
        print(f"[ERROR] 只选中了 {len(selected_indices)} 个点，少于 4 个，跳过该文件", file=sys.stderr)
        return

    # 只取前 4 个点
    selected_indices = selected_indices[:4]

    all_points = np.asarray(pcd.points)
    selected_points = all_points[selected_indices, :]  # 形状 (4, 3)

    # 计算四个点在各轴上的最小值和最大值
    mins = selected_points.min(axis=0)  # [x_min_raw, y_min_raw, z_min_raw]
    maxs = selected_points.max(axis=0)  # [x_max_raw, y_max_raw, z_max_raw]

    # 按你的定义扩展 0.2m
    x_min = mins[0] - 0.2
    x_max = maxs[0] + 0.2
    y_min = mins[1] - 0.2
    y_max = maxs[1] + 0.2
    z_min = mins[2] - 0.2
    z_max = maxs[2] + 0.2

    # 生成保存文件名 (与 PCD 文件同名，改为 txt)
    base_name = os.path.splitext(target_pcd_name)[0]
    save_file = os.path.join(pcd_folder, f"{base_name}.txt")

    with open(save_file, 'w') as f:
        f.write("# 4 selected points (x y z)\n")
        for p in selected_points:
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")

        f.write("# range values in order:\n")
        f.write(f"x_min: {x_min:.1f}\n")
        f.write(f"x_max: {x_max:.1f}\n")
        f.write(f"y_min: {y_min:.1f}\n")
        f.write(f"y_max: {y_max:.1f}\n")
        f.write(f"z_min: {z_min:.1f}\n")
        f.write(f"z_max: {z_max:.1f}\n")

    print(f"[Save] 已保存选点与范围到: {save_file}")
    print("点云处理完成。")

def read_pcd(pcd_dir, ii):
    pcd_fp = os.path.join(pcd_dir, "{:08d}.pcd".format(ii))
    read_pc = o3d.t.io.read_point_cloud(pcd_fp)
    positions = np.asarray(read_pc.point['positions'].cpu().numpy())
    rings = np.asarray(read_pc.point['ring'].cpu().numpy())
    read_pc_np = np.concatenate([positions, rings], axis=-1)
    return read_pc_np

def get_merged_pcd(in_pcd_dir, start, end, out_dir):
    full_pc = None
    for i in range(start, end + 1):
        pc = read_pcd(in_pcd_dir, i)

        if full_pc is None:
            full_pc = pc
        else:
            full_pc = np.concatenate([full_pc, pc], axis=0)
    
    xyz = full_pc[:, :-1]
    pcd = o3d.t.geometry.PointCloud()
    pcd.point["positions"] = o3c.Tensor(xyz, dtype=o3c.Dtype.Float32)

    pcd_path = os.path.join(out_dir, 'merged_pcd.pcd')
    o3d.t.io.write_point_cloud(pcd_path, pcd)

    return pcd_path

def viz_pcd(pcd_path):
    read_pc = o3d.t.io.read_point_cloud(pcd_path)
    read_pc_np = np.asarray(read_pc.point['positions'].cpu().numpy())

    pc2 = o3d.geometry.PointCloud()
    pc2.points = o3d.utility.Vector3dVector(read_pc_np)
    o3d.visualization.draw_geometries([pc2])

# ===================== main =====================

if __name__ == "__main__":
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True, help='pcd (for reading from pcd) or bag (reading from bag)', default='bag')
    parser.add_argument('--out_dir', type=str, required=True, help='path to output dir')
    parser.add_argument('--data_dir', type=str, required=True, 
                            help='path to data dir. for bag mode, path to the ros bag if ROS 1 \
                            and path to directory containing bag/metadata if ROS 2. \
                            for pcd mode, path to directory containing the directory containing pcd files')
    args = parser.parse_args()

    mode = args.mode
    output_dir = args.out_dir
    data_dir = args.data_dir

    if mode == "bag":
        # Mode of reading from bag
        if not os.path.isdir(output_dir):
            print(f"[ERROR] 输出目录 '{output_dir}' 不存在", file=sys.stderr)
            sys.exit(1)

        msg_type = detect_lidar_msg_type(data_dir)
        if msg_type is None:
            print("[ERROR] 未检测到支持的雷达消息类型，退出。", file=sys.stderr)
            sys.exit(1)

        if msg_type == "PointCloud2":
            pcd_path = convert_pointcloud2_bag_to_pcd(
                bag_file=data_dir,
                output_dir=output_dir,
                topic_name="/sensor_scan", # change topic name if different
                pcd_name="sensor_PointCloud2_inten_ascii.pcd"
            )
        else:  # "CustomMsg"
            pcd_path = convert_livox_custom_bag_to_pcd(
                bag_file=data_dir,
                output_dir=output_dir,
                topic_name="/livox/lidar", # change topic name if different
                pcd_name="livox_CustomMsg_inten_ascii.pcd"
            )

        if pcd_path is None:
            print("[ERROR] PCD generation failed", file=sys.stderr)
            sys.exit(1)
    elif mode == "pcd":
        # Mode of reading from PCD files
        in_pcd_dir = os.path.join(data_dir, 'merged_pcd')
        start_frame = 180
        end_frame = 220

        pcd_path = get_merged_pcd(in_pcd_dir, start_frame, end_frame, output_dir)

        # viz_pcd(pcd_path)
    else:
        print('Unknown mode')

    # Interative point selection
    select_and_save_points(
        pcd_folder=output_dir,
        target_pcd_name=os.path.basename(pcd_path)
    )