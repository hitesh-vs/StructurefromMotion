import numpy as np
import cv2
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
import time

def BuildVisibilityMatrix(K, Rset, Cset, Xset):
    """
    Build visibility matrix indicating which 3D points are visible in which cameras.
    
    Parameters:
    - K: Camera intrinsic matrix
    - Rset: List of rotation matrices
    - Cset: List of camera centers
    - Xset: List of 3D point sets visible in each camera
    
    Returns:
    - visibility_matrix: Binary matrix (n_cameras x n_points)
    """
    n_cameras = len(Rset)
    n_points = sum(len(points) for points in Xset)
    
    # Initialize visibility matrix
    visibility_matrix = np.zeros((n_cameras, n_points), dtype=int)
    
    # Set visibility based on Xset
    point_offset = 0
    for cam_idx, points in enumerate(Xset):
        n_cam_points = len(points)
        visibility_matrix[cam_idx, point_offset:point_offset + n_cam_points] = 1
        point_offset += n_cam_points
    
    return visibility_matrix

def get_camera_point_indices(visibility_matrix):
    """Extract camera and point indices from visibility matrix."""
    camera_indices = []
    point_indices = []
    
    n_cameras, n_points = visibility_matrix.shape
    for i in range(n_cameras):
        for j in range(n_points):
            if visibility_matrix[i, j] == 1:
                camera_indices.append(i)
                point_indices.append(j)
                
    return np.array(camera_indices), np.array(point_indices)

def build_jacobian_sparsity(n_cameras, n_points, camera_indices, point_indices):
    """Create sparse matrix for the Jacobian structure."""
    m = len(camera_indices) * 2  # Each observation contributes 2 residuals (x,y)
    n = n_cameras * 6 + n_points * 3  # 6 params per camera, 3 per point
    A = lil_matrix((m, n), dtype=int)
    
    i = np.arange(len(camera_indices))
    
    # For each observation, mark dependencies on camera parameters
    for s in range(6):
        A[2 * i, camera_indices * 6 + s] = 1
        A[2 * i + 1, camera_indices * 6 + s] = 1
    
    # For each observation, mark dependencies on 3D point parameters
    for s in range(3):
        A[2 * i, n_cameras * 6 + point_indices * 3 + s] = 1
        A[2 * i + 1, n_cameras * 6 + point_indices * 3 + s] = 1
    
    return A

def reprojection_error(params, n_cameras, n_points, K, camera_indices, point_indices, observations):
    """
    Compute reprojection error for all observations.
    
    Parameters:
    - params: Combined array of camera parameters and 3D point coordinates
    - n_cameras: Number of cameras
    - n_points: Number of 3D points
    - K: Camera intrinsic matrix
    - camera_indices: Camera index for each observation
    - point_indices: Point index for each observation
    - observations: 2D pixel coordinates of observed points
    
    Returns:
    - residuals: Flattened array of reprojection errors (x and y components)
    """
    # Reshape parameters
    camera_params = params[:n_cameras * 6].reshape((n_cameras, 6))
    points_3d = params[n_cameras * 6:].reshape((n_points, 3))
    
    # Initialize residuals array
    residuals = np.zeros(len(camera_indices) * 2)
    
    # Compute residuals for each observation
    for i, (cam_idx, point_idx) in enumerate(zip(camera_indices, point_indices)):
        # Extract camera parameters
        rvec = camera_params[cam_idx, :3].reshape(3, 1)
        C = camera_params[cam_idx, 3:6]
        
        # Convert rotation vector to matrix
        R, _ = cv2.Rodrigues(rvec)
        
        # Compute translation vector t = -RC
        t = -R @ C.reshape(3, 1)
        
        # Create projection matrix P = K[R|t]
        P = K @ np.hstack([R, t])
        
        # Project 3D point to 2D
        X = points_3d[point_idx]
        X_homog = np.append(X, 1)
        x_proj_homog = P @ X_homog
        
        # Convert from homogeneous coordinates
        x_proj = x_proj_homog[:2] / x_proj_homog[2] if x_proj_homog[2] != 0 else x_proj_homog[:2]
        
        # Calculate residual
        residuals[2*i:2*i+2] = observations[i] - x_proj
    
    return residuals

def BundleAdjustment(K, Rset, Cset, points_3d, points_2d, visibility_matrix, 
                    method='trf', ftol=1e-8, max_iterations=1000, 
                    outlier_threshold=None, regularization_weight=None):
    """
    Main function to perform bundle adjustment.
    
    Parameters:
    - K: Fixed camera intrinsic matrix
    - Rset: List of rotation matrices
    - Cset: List of camera centers
    - points_3d: List of 3D points
    - points_2d: List of lists, where points_2d[i] contains the 2D points visible in camera i
    - visibility_matrix: Binary matrix indicating which points are visible from which cameras
    - method: Optimization method ('trf', 'dogbox', or 'lm')
    - ftol: Tolerance for termination by the change of the cost function
    - max_iterations: Maximum number of function evaluations
    - outlier_threshold: Threshold for outlier rejection (in pixels)
    - regularization_weight: Weight for regularization term
    
    Returns:
    - refined_points: Refined 3D points
    - refined_Rset: Refined rotation matrices
    - refined_Cset: Refined camera centers
    """
    print("Running Bundle Adjustment...")
    start_time = time.time()
    
    # Convert inputs to numpy arrays if needed
    points_3d = np.asarray(points_3d)
    visibility_matrix = np.asarray(visibility_matrix)
    
    n_cameras = len(Rset)
    n_points = len(points_3d)
    
    print(f"Number of cameras: {n_cameras}")
    print(f"Number of 3D points: {n_points}")
    
    # Generate camera indices, point indices, and observations
    camera_indices, point_indices = get_camera_point_indices(visibility_matrix)
    
    # Prepare 2D observations array
    observations = []
    valid_camera_indices = []
    valid_point_indices = []
    
    for i, cam_idx in enumerate(camera_indices):
        point_idx = point_indices[i]
        
        # Check if we have valid 2D points for this camera
        if points_2d[cam_idx] is not None and point_idx < len(points_2d[cam_idx]):
            observations.append(points_2d[cam_idx][point_idx]) 
            valid_camera_indices.append(cam_idx)
            valid_point_indices.append(point_idx)
    
    # Update camera_indices and point_indices with valid ones
    camera_indices = np.array(valid_camera_indices)
    point_indices = np.array(valid_point_indices)
    observations = np.array(observations)
    
    if len(observations) == 0:
        print("No valid observations found. Cannot perform bundle adjustment.")
        return points_3d, Rset, Cset
    
    print(f"Using {len(observations)} valid observations for bundle adjustment")
    
    # Convert rotation matrices to Rodriguez vectors
    camera_params = []
    for R, C in zip(Rset, Cset):
        # Convert rotation matrix to Rodriguez vector
        rvec, _ = cv2.Rodrigues(R)
        params = np.concatenate([rvec.flatten(), C.flatten()])
        camera_params.append(params)
    
    camera_params = np.array(camera_params)
    
    # Initial parameter vector
    x0 = np.hstack((camera_params.ravel(), points_3d.ravel()))
    
    # Build Jacobian sparsity pattern
    A = build_jacobian_sparsity(n_cameras, n_points, camera_indices, point_indices)
    print(f"Jacobian sparsity matrix shape: {A.shape}")
    
    # Define objective function
    def objective(params):
        residuals = reprojection_error(
            params, n_cameras, n_points, K, 
            camera_indices, point_indices, observations
        )
        
        # Optional outlier rejection
        if outlier_threshold:
            # Reshape to get per-point errors
            point_errors = np.sqrt(
                residuals.reshape(-1, 2)[:, 0]**2 + 
                residuals.reshape(-1, 2)[:, 1]**2
            )
            
            # Apply robust weighting (Huber loss)
            weights = np.ones_like(point_errors)
            outlier_mask = point_errors > outlier_threshold
            weights[outlier_mask] = outlier_threshold / point_errors[outlier_mask]
            
            # Apply weights to residuals
            residuals = residuals.reshape(-1, 2) * weights[:, np.newaxis]
            residuals = residuals.ravel()
        
        # Optional regularization
        if regularization_weight:
            # Regularize camera positions to prevent drift
            camera_reg = regularization_weight * params[:n_cameras * 6].reshape(n_cameras, 6)[:, 3:6].ravel()
            # Regularize 3D point positions
            point_reg = regularization_weight * params[n_cameras * 6:].ravel()
            
            # Concatenate all terms
            return np.concatenate([residuals, camera_reg, point_reg])
        
        return residuals
    
    # Run optimization
    print("Starting optimization...")
    result = least_squares(
        objective, x0, 
        jac_sparsity=A, 
        verbose=2, 
        x_scale='jac',
        ftol=ftol,
        max_nfev=max_iterations, 
        method=method
    )
    
    # Extract optimized parameters
    refined_params = result.x
    refined_camera_params = refined_params[:n_cameras * 6].reshape(n_cameras, 6)
    refined_points = refined_params[n_cameras * 6:].reshape(n_points, 3)
    
    # Convert back to rotation matrices and camera centers
    refined_Rset = []
    refined_Cset = []
    for params in refined_camera_params:
        rvec = params[:3].reshape(3, 1)
        C = params[3:6]
        R, _ = cv2.Rodrigues(rvec)
        refined_Rset.append(R)
        refined_Cset.append(C)
    
    # Calculate final error
    final_residuals = reprojection_error(
        refined_params, n_cameras, n_points, K,
        camera_indices, point_indices, observations
    )
    final_error = np.sqrt(np.mean(final_residuals**2))
    
    end_time = time.time()
    print(f"Bundle adjustment completed in {end_time - start_time:.2f} seconds")
    print(f"Final reprojection error (RMSE): {final_error:.4f} pixels")
    
    return refined_points, refined_Rset, refined_Cset
