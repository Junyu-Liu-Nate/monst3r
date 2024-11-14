import numpy as np
import pandas as pd
from pyntcloud import PyntCloud
import os
import cv2

def quaternion_to_rotation_matrix(qw, qx, qy, qz):
    # Compute rotation matrix from quaternion
    R = np.array([
        [1 - 2*(qy**2 + qz**2),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [    2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2),     2*(qy*qz - qx*qw)],
        [    2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
    ])
    return R

def read_point_cloud(file_path):
    # Read point cloud from file using PyntCloud
    pc = PyntCloud.from_file(file_path)
    points = pc.points[['x', 'y', 'z']].values
    if {'red', 'green', 'blue'}.issubset(pc.points.columns):
        colors = pc.points[['red', 'green', 'blue']].values / 255.0
    else:
        colors = np.zeros((points.shape[0], 3))
    return points, colors

def write_point_cloud(file_path, points, colors):
    # Write point cloud to file using PyntCloud
    data = pd.DataFrame({
        'x': points[:, 0],
        'y': points[:, 1],
        'z': points[:, 2],
        'red': (colors[:, 0] * 255).astype(np.uint8),
        'green': (colors[:, 1] * 255).astype(np.uint8),
        'blue': (colors[:, 2] * 255).astype(np.uint8)
    })
    pc = PyntCloud(data)
    pc.to_file(file_path)

# Paths to data
data_dir = os.path.join('demo_tmp', 'lady-running-65-224')
pointcloud_template = os.path.join(data_dir, 'pointcloud_{}.ply')
mask_template = os.path.join(data_dir, 'enlarged_dynamic_mask_{}.png')
intrinsics_file = os.path.join(data_dir, 'pred_intrinsics.txt')
trajectory_file = os.path.join(data_dir, 'pred_traj.txt')

# Load camera intrinsics per frame
intrinsics_list = []
with open(intrinsics_file, 'r') as f:
    for line in f:
        values = np.fromstring(line, sep=' ')
        K = values.reshape(3, 3)
        intrinsics_list.append(K)

# Load camera poses per frame
poses_list = []
with open(trajectory_file, 'r') as f:
    for line in f:
        t, x, y, z, qw, qx, qy, qz = map(float, line.strip().split())
        R = quaternion_to_rotation_matrix(qw, qx, qy, qz)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [x, y, z]
        poses_list.append(T)

# Initialize lists for combined static point cloud
static_points_combined = []
static_colors_combined = []

# Process each frame
for i in range(len(poses_list)):
    # Load point cloud (points are in world coordinates)
    points, colors = read_point_cloud(pointcloud_template.format(i))

    # Load mask image
    mask = cv2.imread(mask_template.format(i), cv2.IMREAD_GRAYSCALE)
    height, width = mask.shape

    # Transform points from world to camera coordinates
    points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))  # (N, 4)
    camera_pose_inv = np.linalg.inv(poses_list[i])  # World to camera transformation
    pts_cam = (camera_pose_inv @ points_homogeneous.T).T[:, :3]  # (N, 3)

    # Only consider points in front of the camera
    valid_depth = pts_cam[:, 2] > 0
    pts_cam = pts_cam[valid_depth]
    colors = colors[valid_depth]
    points_world = points[valid_depth]

    # Project into image plane
    K = intrinsics_list[i]
    pixels_homogeneous = K @ pts_cam.T  # (3, N)
    pixels = (pixels_homogeneous[:2, :] / pixels_homogeneous[2, :]).T  # (N, 2)
    pixels = pixels.astype(int)

    # Filter valid pixels within image boundaries
    valid_pixels = (
        (pixels[:, 0] >= 0) & (pixels[:, 0] < width) &
        (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
    )
    pixels = pixels[valid_pixels]
    points_world = points_world[valid_pixels]
    colors = colors[valid_pixels]

    # Separate dynamic and static points using mask
    mask_values = mask[pixels[:, 1], pixels[:, 0]]
    dynamic = mask_values > 0
    static = ~dynamic

    # Dynamic points
    if np.any(dynamic):
        dyn_points = points_world[dynamic]
        dyn_colors = colors[dynamic]
        write_point_cloud(os.path.join(data_dir, f'dynamic_points_{i}.ply'), dyn_points, dyn_colors)

    # Static points
    if np.any(static):
        stat_points = points_world[static]
        stat_colors = colors[static]
        static_points_combined.append(stat_points)
        static_colors_combined.append(stat_colors)

# Save combined static point cloud
if static_points_combined:
    static_points_combined = np.vstack(static_points_combined)
    static_colors_combined = np.vstack(static_colors_combined)
    write_point_cloud(os.path.join(data_dir, 'static_combined.ply'), static_points_combined, static_colors_combined)
else:
    print("No static points found.")