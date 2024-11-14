import open3d as o3d
import numpy as np
import os
import sys
from scipy.spatial.transform import Rotation as R

repo_dir = "/Users/liujunyu/Desktop/Course/Brown/CSCI2951I/code/monst3r"

#%% File I/O
def load_intrinsics(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    intrinsics = []
    for line in lines:
        intrinsics.append(np.array(line.strip().split(), dtype=float).reshape(3, 3))
    return intrinsics

def load_trajectories(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    trajectories = []
    for line in lines:
        data = list(map(float, line.strip().split()))
        t = data[0]
        translation = np.array(data[1:4])
        rotation = np.array(data[4:])
        trajectories.append((t, translation, rotation))
    return trajectories

def get_frame_info(intrinsics, trajectories, frame_number):
    intrinsic = intrinsics[frame_number]
    trajectory = trajectories[frame_number]
    return intrinsic, trajectory

#%% Render with open3dSpecify camera views and intrinsic parameters
def render_open3d():
    frame_number = 24

    # pc_path = os.path.join('demo_tmp', 'lady-running-65-224', f'pointcloud_{frame_number}.ply')
    # pc_path = os.path.join('demo_tmp', 'lady-running-65-224', f'dynamic_points_{frame_number}.ply')
    pc_path = os.path.join('demo_tmp', 'lady-running-65-224', f'static_combined.ply')
    if not os.path.exists(pc_path):
        print(f"Point cloud file does not exist: {pc_path}")
        sys.exit(1)
        
    pcd = o3d.io.read_point_cloud(pc_path)
    if not pcd.has_points():
        print(f"Point cloud has no points: {pc_path}")
        sys.exit(1)

    intrinsics_path = os.path.join(repo_dir, 'demo_tmp', 'lady-running-65-224', 'pred_intrinsics.txt')
    trajectories_path = os.path.join(repo_dir, 'demo_tmp', 'lady-running-65-224', 'pred_traj.txt')
    
    if not os.path.exists(intrinsics_path):
        print(f"Intrinsics file does not exist: {intrinsics_path}")
        sys.exit(1)
        
    if not os.path.exists(trajectories_path):
        print(f"Trajectories file does not exist: {trajectories_path}")
        sys.exit(1)
    
    intrinsics = load_intrinsics(intrinsics_path)
    trajectories = load_trajectories(trajectories_path)
    
    if frame_number >= len(intrinsics):
        print(f"Frame number {frame_number} exceeds intrinsics data length {len(intrinsics)}")
        sys.exit(1)
        
    if frame_number >= len(trajectories):
        print(f"Frame number {frame_number} exceeds trajectories data length {len(trajectories)}")
        sys.exit(1)
    
    intrinsic, trajectory = get_frame_info(intrinsics, trajectories, frame_number)
    
    print(pcd)
    print(f"Intrinsic parameters for frame {frame_number}: \n{intrinsic}")
    print(f"Trajectory for frame {frame_number}: \nTranslation: {trajectory[1]}\nRotation: {trajectory[2]}")
    
    # Minimal Visualization Test
    try:
        print("Starting minimal visualization test...")
        o3d.visualization.draw_geometries([pcd])
        print("Minimal visualization succeeded.")
    except Exception as e:
        print(f"Minimal visualization error: {e}")
        sys.exit(1)
    
    # If minimal visualization works, proceed with camera parameters
    try:
        print("Initializing visualizer with camera parameters...")
        # Initialize the visualizer
        vis = o3d.visualization.Visualizer()
        print("Creating window...")
        vis.create_window()
        print("Adding geometry to visualizer...")
        vis.add_geometry(pcd)
        print("Geometry added.")

        # Set the camera intrinsic parameters
        print("Setting camera intrinsics...")
        intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic()
        intrinsic_o3d.set_intrinsics(
            width=224,  # Replace with your image width
            height=224, # Replace with your image height
            fx=intrinsic[0, 0],
            fy=intrinsic[1, 1],
            cx=intrinsic[0, 2],
            cy=intrinsic[1, 2]
        )
        print(f"Camera intrinsics set: \n{intrinsic_o3d.intrinsic_matrix}")

        # Convert quaternion from [qw, qx, qy, qz] to [qx, qy, qz, qw]
        qw, qx, qy, qz = trajectory[2]
        quaternion = [qx, qy, qz, qw]
        print(f"Quaternion (x, y, z, w): {quaternion}")
        
        # Check if quaternion is normalized
        norm = np.linalg.norm(quaternion)
        print(f"Quaternion norm: {norm}")
        if not np.isclose(norm, 1.0, atol=1e-6):
            print("Quaternion is not normalized. Normalizing now.")
            quaternion = [q / norm for q in quaternion]
            print(f"Normalized Quaternion: {quaternion}")
        
        # Check for NaN or Inf in quaternion
        if any(np.isnan(quaternion)) or any(np.isinf(quaternion)):
            print("Invalid quaternion: contains NaN or Inf.")
            sys.exit(1)

        # # Convert to rotation matrix using Open3D
        # try:
        #     print(f"Converting quaternion {trajectory[2]} to rotation matrix with Open3D...")
        #     rotation_matrix = o3d.geometry.get_rotation_matrix_from_quaternion(quaternion)
        #     print(f"Rotation matrix from Open3D: \n{rotation_matrix}")
        # except Exception as e:
        #     print(f"Error converting quaternion with Open3D: {e}")
        #     sys.exit(1)
        # except:
        #     print("Segmentation fault occurred during quaternion conversion with Open3D.")
        #     sys.exit(1)
        
        # Alternatively, use SciPy to get rotation matrix
        try:
            print("Converting quaternion to rotation matrix with SciPy...")
            rotation = R.from_quat(quaternion)  # [x, y, z, w]
            rotation_matrix_scipy = rotation.as_matrix()
            print(f"Rotation matrix from SciPy: \n{rotation_matrix_scipy}")
            # Use SciPy's rotation matrix
            rotation_matrix = rotation_matrix_scipy
        except Exception as e:
            print(f"Error converting quaternion with SciPy: {e}")
        
        # Set the camera extrinsic parameters
        print("Setting camera extrinsics...")
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = rotation_matrix
        extrinsic[:3, 3] = trajectory[1]
        print(f"Extrinsic matrix: \n{extrinsic}")
        
        # Create camera parameters
        print("Creating camera parameters...")
        camera_params = o3d.camera.PinholeCameraParameters()
        camera_params.intrinsic = intrinsic_o3d
        camera_params.extrinsic = extrinsic
        print("Camera parameters created.")

        # Apply the camera parameters to the visualizer
        print("Applying camera parameters to visualizer...")
        ctr = vis.get_view_control()
        ctr.convert_from_pinhole_camera_parameters(camera_params)
        print("Camera parameters applied.")

        # Update the visualizer to apply the camera settings
        print("Updating visualizer...")
        vis.poll_events()
        vis.update_renderer()
        print("Visualizer updated.")

        print("Running visualizer...")
        vis.run()
        print("Visualizer run completed.")
        
        # Print out the camera view information
        print("Camera intrinsic parameters:")
        print(camera_params.intrinsic)
        print("Camera extrinsic parameters:")
        print(camera_params.extrinsic)
        
    except Exception as e:
        print(f"Visualization with camera parameters error: {e}")
    finally:
        # Destroy the visualizer window
        vis.destroy_window()
        print("Visualizer window destroyed.")

if __name__ == "__main__":
    render_open3d()