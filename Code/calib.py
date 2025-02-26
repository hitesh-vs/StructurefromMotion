import numpy as np
import cv2
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
import time

def BundleAdjustment(K, Rset, Cset, points_3d, points_2d, visibility_matrix):
    """
    Main function to perform bundle adjustment.
    
    Parameters:
    - K: Fixed camera intrinsic matrix
    - Rset: List of rotation matrices
    - Cset: List of camera centers
    - points_3d: List of 3D points (size n_points)
    - points_2d: List of corresponding 2D reference points (size n_points)
    - visibility_matrix: Binary matrix (n_cameras x n_points) indicating which points are visible from which cameras
    
    Returns:
    - refined_points: Refined 3D points
    - refined_Rset: Refined rotation matrices
    - refined_Cset: Refined camera centers
    """
    print("Running Bundle Adjustment...")
    
    # Convert inputs to numpy arrays if they aren't already
    points_3d = np.asarray(points_3d)
    points_2d = np.asarray(points_2d)
    visibility_matrix = np.asarray(visibility_matrix)
    
    n_cameras = len(Rset)
    n_points = len(points_3d)
    
    print(f"Number of cameras: {n_cameras}")
    print(f"Number of 3D points: {n_points}")
    
    # Generate camera indices, point indices, and observations from visibility matrix
    camera_indices = []
    point_indices = []
    observations = []
    
    # For each point and each camera that sees it
    for j in range(n_points):  # For each 3D point
        for i in range(n_cameras):  # For each camera
            if visibility_matrix[i, j] == 1:  # If camera i sees point j
                camera_indices.append(i)
                point_indices.append(j)
                observations.append(points_2d[j])  # Use the reference 2D point
    
    camera_indices = np.array(camera_indices)
    point_indices = np.array(point_indices)
    observations = np.array(observations)
    
    print(f"Generated {len(camera_indices)} observations from visibility matrix")
    
    # Run the bundle adjustment
    refined_points, refined_Rset, refined_Cset = bundle_adjustment(
        points_3d, observations, K, Rset, Cset,
        camera_indices, point_indices, visibility_matrix
    )
    
    print("Bundle Adjustment completed successfully.")
    
    return refined_points, refined_Rset, refined_Cset

def bundle_adjustment(points_3d, observations, K, Rset, Cset, camera_indices, point_indices, visibility):
    """
    Perform bundle adjustment to refine camera poses and 3D points with fixed intrinsics K.
    
    Parameters:
    - points_3d: Array of 3D point coordinates (n_points, 3)
    - observations: Array of 2D observations (n_observations, 2)
    - K: Fixed camera intrinsic matrix
    - Rset: List of rotation matrices
    - Cset: List of camera centers
    - camera_indices: Array mapping each observation to a camera
    - point_indices: Array mapping each observation to a 3D point
    - visibility: Visibility matrix (n_cameras x n_points)
    
    Returns:
    - refined_points_3d: Refined 3D points
    - refined_Rset: Refined rotation matrices
    - refined_Cset: Refined camera centers
    """
    start_time = time.time()
    
    # Number of cameras and points
    n_cameras = len(Rset)
    n_points = points_3d.shape[0]
    n_observations = observations.shape[0]
    
    # Extract camera parameters (rotation R, camera center C)
    camera_params = []
    for R, C in zip(Rset, Cset):
        # Convert rotation matrix to Rodriguez vector (3 parameters)
        rvec, _ = cv2.Rodrigues(R)
        # Camera parameters: 3 for rotation, 3 for camera center
        params = np.concatenate([rvec.flatten(), C.flatten()])
        camera_params.append(params)
    
    camera_params = np.array(camera_params)
    
    # Define the parameter vector for optimization (camera parameters + 3D points)
    x0 = np.hstack((camera_params.flatten(), points_3d.flatten()))
    
    # Define the sparsity structure of the Jacobian matrix
    A = build_jacobian_sparsity(n_cameras, n_points, camera_indices, point_indices)
    
    # Define the objective function (sum of squared reprojection errors)
    def objective(params):
        return reprojection_error(params, n_cameras, n_points, K, camera_indices, 
                                 point_indices, observations, visibility)
    
    # Perform the optimization
    print(f"Starting optimization with {n_cameras} cameras and {n_points} points...")
    result = least_squares(objective, x0, jac_sparsity=A, verbose=2, x_scale='jac',
                          method='trf', ftol=1e-5, max_nfev=100)
    
    # Extract the refined parameters
    refined_params = result.x
    refined_camera_params = refined_params[:n_cameras * 6].reshape(n_cameras, 6)
    refined_points_3d = refined_params[n_cameras * 6:].reshape(n_points, 3)
    
    end_time = time.time()
    print(f"Bundle adjustment completed in {end_time - start_time:.2f} seconds")
    print(f"Initial cost: {np.sum(objective(x0)**2):.6f}")
    print(f"Final cost: {np.sum(objective(refined_params)**2):.6f}")
    
    # Convert refined camera parameters back to R, C form
    refined_Rset = []
    refined_Cset = []
    for params in refined_camera_params:
        rvec = params[:3].reshape(3, 1)
        C = params[3:6]
        R, _ = cv2.Rodrigues(rvec)
        refined_Rset.append(R)
        refined_Cset.append(C)
    
    return refined_points_3d, refined_Rset, refined_Cset

def build_jacobian_sparsity(n_cameras, n_points, camera_indices, point_indices):
    """
    Create a sparse matrix defining the structure of the Jacobian matrix.
    This is used to exploit sparsity for faster optimization.
    """
    m = len(camera_indices) * 2  # Each 2D point contributes 2 residuals (x and y)
    n = n_cameras * 6 + n_points * 3  # 6 params per camera (3 for R, 3 for C), 3 coords per 3D point
    A = lil_matrix((m, n), dtype=int)
    
    i = np.arange(len(camera_indices))
    # For each observation, the residuals depend on the corresponding camera parameters
    for s in range(6):
        A[2 * i, camera_indices * 6 + s] = 1
        A[2 * i + 1, camera_indices * 6 + s] = 1
    
    # For each observation, the residuals also depend on the corresponding 3D point
    for s in range(3):
        A[2 * i, n_cameras * 6 + point_indices * 3 + s] = 1
        A[2 * i + 1, n_cameras * 6 + point_indices * 3 + s] = 1
    
    return A

def reprojection_error(params, n_cameras, n_points, K, camera_indices, point_indices, observations, visibility):
    """
    Compute the reprojection error for each observation using fixed K.
    
    This is the objective function to minimize.
    """
    # Extract camera parameters and 3D points from the parameter vector
    camera_params = params[:n_cameras * 6].reshape(n_cameras, 6)
    points_3d = params[n_cameras * 6:].reshape(n_points, 3)
    
    # Initialize residuals
    residuals = np.zeros(len(camera_indices) * 2)
    
    # For each observation
    for i, (cam_idx, point_idx) in enumerate(zip(camera_indices, point_indices)):
        # Get camera parameters
        rvec = camera_params[cam_idx, :3].reshape(3, 1)
        C = camera_params[cam_idx, 3:6].reshape(3, 1)
        
        # Convert rvec to rotation matrix
        R, _ = cv2.Rodrigues(rvec)
        
        # Compute translation vector from R and C
        t = -R @ C
        
        # Project the 3D point to 2D
        X = points_3d[point_idx]
        X_homogeneous = np.append(X, 1)
        
        # Create projection matrix
        P = K @ np.hstack([R, t])
        
        # Project the point
        x_proj = P @ X_homogeneous
        x_proj = x_proj[:2] / x_proj[2]  # Normalize by dividing by z
        
        # Compute residuals (difference between observed and projected points)
        residuals[2*i:2*i+2] = observations[i] - x_proj
    
    return residuals
