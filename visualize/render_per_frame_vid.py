import numpy as np
import pandas as pd
from pyntcloud import PyntCloud
import os
import cv2
import pyvista as pv

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

def create_camera_traj_rotate(initial_pose, num_frames):
    """Create a camera trajectory that rotates the camera around X, Y, and Z axes and returns to the initial position."""
    camera_poses = []
    for i in range(num_frames):
        # Compute small oscillation angles that start and end at 0
        angle_x = np.sin(np.pi * i / num_frames) * 0.05  # Rotation around X-axis
        angle_y = np.sin(np.pi * i / num_frames) * 0.1  # Rotation around Y-axis
        angle_z = np.sin(np.pi * i / num_frames) * 0.05 # Rotation around Z-axis

        R = initial_pose[:3, :3]
        t = initial_pose[:3, 3]

        # Rotation matrices for X, Y, Z
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(angle_x), -np.sin(angle_x)],
            [0, np.sin(angle_x), np.cos(angle_x)]
        ])
        Ry = np.array([
            [np.cos(angle_y), 0, np.sin(angle_y)],
            [0, 1, 0],
            [-np.sin(angle_y), 0, np.cos(angle_y)]
        ])
        Rz = np.array([
            [np.cos(angle_z), -np.sin(angle_z), 0],
            [np.sin(angle_z),  np.cos(angle_z), 0],
            [0, 0, 1]
        ])

        # Combine rotations
        new_R = R @ (Rx @ Ry @ Rz)
        # new_R = R @ (Rz)
        new_pose = np.eye(4)
        new_pose[:3, :3] = new_R
        new_pose[:3, 3] = t  # Keep the same position
        camera_poses.append(new_pose)
    return camera_poses

def create_camera_traj_zoom(initial_pose, num_frames):
    """
    Create a camera trajectory that smoothly zooms the camera in and out along its local Z-axis,
    starting and ending at the initial position.
    """
    camera_poses = []
    R = initial_pose[:3, :3]  # Camera rotation matrix (orientation)
    t = initial_pose[:3, 3]   # Camera translation vector (position)

    max_distance = 0.1  # Maximum zoom distance (negative to move forward)
    min_distance = 0.0   # No backward movement beyond initial position

    for i in range(num_frames):
        # Parameter s varies from 0 to 1 and back to 0 over num_frames
        s = 0.5 * (1 - np.cos(2 * np.pi * i / (num_frames - 1)))  # Cosine function for smoothness

        # Compute the translation distance along Z-axis
        translation_distance = s * (max_distance - min_distance) + min_distance  # Varies smoothly

        # Move along the camera's local Z-axis
        delta_t = R @ np.array([0, 0, translation_distance])  # Transform to world coordinates
        new_t = t + delta_t  # Updated camera position

        # Create new pose with the same orientation but updated position
        new_pose = np.eye(4)
        new_pose[:3, :3] = R  # Keep the same rotation (orientation)
        new_pose[:3, 3] = new_t  # Updated translation (position)
        camera_poses.append(new_pose)
    return camera_poses

def create_camera_traj_move(initial_pose, num_frames):
    """
    Create a camera trajectory that moves the camera in a circular path around a center point
    in the camera's local X-Y plane, starting and ending at the initial position,
    without changing the camera's orientation.
    """
    camera_poses = []
    R = initial_pose[:3, :3]  # Camera rotation matrix (orientation)
    t = initial_pose[:3, 3]   # Camera translation vector (position)

    radius = 0.03  # Adjust radius as needed for subtle movement

    # Define the center in local coordinates by shifting along the X-axis
    center_shift_x = -radius  # Negative to keep the initial position at the starting point
    center_local = np.array([center_shift_x, 0, 0])

    for i in range(num_frames):
        # Angle varies from 0 to 2π over num_frames - 1 frames
        angle = 2 * np.pi * i / (num_frames - 1)

        # Compute position in local coordinates
        position_local = center_local + np.array([
            radius * np.cos(angle),
            radius * np.sin(angle),
            0
        ])

        # Transform to world coordinates
        position_world = R @ position_local + t

        # Create new pose with the same orientation but updated position
        new_pose = np.eye(4)
        new_pose[:3, :3] = R  # Keep the same rotation (orientation)
        new_pose[:3, 3] = position_world  # Updated translation (position)
        camera_poses.append(new_pose)
    return camera_poses

def render_frame(points, colors, camera_pose, K, image_size):
    # Transform points to camera coordinates
    points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))  # (N, 4)
    camera_pose_inv = np.linalg.inv(camera_pose)
    pts_cam = (camera_pose_inv @ points_homogeneous.T).T[:, :3]

    # Only consider points in front of the camera
    valid_depth = pts_cam[:, 2] > 0
    pts_cam = pts_cam[valid_depth]
    colors = colors[valid_depth]

    # Project into image plane
    pixels_homogeneous = K @ pts_cam.T  # (3, N)
    pixels = (pixels_homogeneous[:2, :] / pixels_homogeneous[2, :]).T  # (N, 2)
    pixels = np.round(pixels).astype(int)

    height, width = image_size
    # Filter valid pixels within image boundaries
    valid_pixels = (
        (pixels[:, 0] >= 0) & (pixels[:, 0] < width) &
        (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
    )
    pixels = pixels[valid_pixels]
    colors = colors[valid_pixels]

    # Swap color channels from RGB to BGR
    colors_bgr = colors[:, [2, 1, 0]]  # Swap red and blue channels

    # Create image and set pixel colors
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[pixels[:, 1], pixels[:, 0]] = (colors_bgr * 255).astype(np.uint8)

    return image

# def render_frame(points, colors, camera_pose, K, image_size):
#     # Transform points to camera coordinates
#     points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))  # (N, 4)
#     camera_pose_inv = np.linalg.inv(camera_pose)
#     pts_cam = (camera_pose_inv @ points_homogeneous.T).T[:, :3]

#     # Only consider points in front of the camera
#     valid_depth = pts_cam[:, 2] > 0
#     pts_cam = pts_cam[valid_depth]
#     colors = colors[valid_depth]

#     # Project into image plane
#     pixels_homogeneous = K @ pts_cam.T  # (3, N)
#     pixels = (pixels_homogeneous[:2, :] / pixels_homogeneous[2, :]).T  # (N, 2)

#     # Round pixel coordinates to integer values
#     pixels = np.round(pixels).astype(int)

#     height, width = image_size

#     # Filter valid pixels within image boundaries
#     valid_pixels = (
#         (pixels[:, 0] >= 0) & (pixels[:, 0] < width) &
#         (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
#     )

#     # Apply valid pixel mask to all arrays
#     pixels = pixels[valid_pixels]
#     colors = colors[valid_pixels]
#     depths = pts_cam[valid_pixels, 2]

#     # Initialize image and depth buffer
#     image = np.zeros((height, width, 3), dtype=np.uint8)
#     depth_buffer = np.full((height, width), np.inf)

#     # Flatten pixel indices for easier indexing
#     pixel_indices = pixels[:, 1] * width + pixels[:, 0]

#     # Sort points by depth (closest first)
#     sorted_indices = np.argsort(depths)

#     # Iterate over points in order from closest to farthest
#     for idx in sorted_indices:
#         x, y = pixels[idx]
#         depth = depths[idx]
#         color = (colors[idx] * 255).astype(np.uint8)

#         # If this point is closer than the current depth buffer value
#         if depth < depth_buffer[y, x]:
#             depth_buffer[y, x] = depth
#             image[y, x] = color

#     return image

def main():
    # Paths to data
    data_dir = os.path.join('demo_tmp', 'som_vid', 'breakdance-flare')
    static_pointcloud_file = os.path.join(data_dir, 'static_combined.ply')
    intrinsics_file = os.path.join(data_dir, 'pred_intrinsics.txt')
    trajectory_file = os.path.join(data_dir, 'pred_traj.txt')
    mask_template = os.path.join(data_dir, 'enlarged_dynamic_mask_{}.png')

    selected_frame = 1

    # Load camera intrinsics
    intrinsics_list = []
    with open(intrinsics_file, 'r') as f:
        for line in f:
            values = np.fromstring(line, sep=' ')
            K = values.reshape(3, 3)
            intrinsics_list.append(K)
    # Use the selected frame's intrinsics
    K = intrinsics_list[selected_frame]

    # Load camera poses
    poses_list = []
    with open(trajectory_file, 'r') as f:
        for line in f:
            t, x, y, z, qw, qx, qy, qz = map(float, line.strip().split())
            R = quaternion_to_rotation_matrix(qw, qx, qy, qz)
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = [x, y, z]
            poses_list.append(T)
    # Use the first frame's pose as initial pose
    initial_pose = poses_list[selected_frame]

    # Load the static point cloud
    points, colors = read_point_cloud(static_pointcloud_file)

    # Get image size from the first mask
    mask = cv2.imread(mask_template.format(0), cv2.IMREAD_GRAYSCALE)
    if mask is not None:
        height, width = mask.shape
    else:
        # Set default image size
        height, width = 480, 640
    image_size = (height, width)

    # Define camera trajectory
    num_frames = 60  # Number of frames in the video
    # camera_poses = create_camera_traj_rotate(initial_pose, num_frames)
    # camera_poses = create_camera_traj_zoom(initial_pose, num_frames)
    camera_poses = create_camera_traj_move(initial_pose, num_frames)

    # Output directory for frames
    output_dir = os.path.join(data_dir, 'rendered_frames')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Render frames
    for i, camera_pose in enumerate(camera_poses):
        print(f'Rendering frame {i+1}/{num_frames}')
        image = render_frame(points, colors, camera_pose, K, image_size)
        cv2.imwrite(os.path.join(output_dir, f'frame_{i:04d}.png'), image)

    # Create video from frames
    video_filename = os.path.join(data_dir, 'rendered_video_' + str(selected_frame) + '.mp4')
    frame_rate = 30  # Frames per second

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_filename, fourcc, frame_rate, (width, height))

    for i in range(num_frames):
        frame_path = os.path.join(output_dir, f'frame_{i:04d}.png')
        frame = cv2.imread(frame_path)
        video_writer.write(frame)

    video_writer.release()
    print(f'Video saved to {video_filename}')

if __name__ == '__main__':
    main()