import numpy as np
import numpy as np

import numpy as np

# def read_matches_file(filename, image_id1, image_id2):
#     """
#     Parse matches.txt file to extract corresponding points between two specific images.
#     """
#     points1 = []
#     points2 = []

    
#     with open(filename, 'r') as f:
#         # Read and process all lines
#         lines = f.readlines()
        
#         # Find the first non-empty line
#         n_features = 0
#         for line in lines:
#             line = line.strip()
#             if line:  # if line is not empty
#                 if 'nFeatures' in line:
#                     n_features = int(line.split(': ')[1])
#                     break
        
#         # Process each feature line
#         current_feature = 0
#         line_idx = 1  # Start from next line after nFeatures
        
#         while current_feature < n_features and line_idx < len(lines):
#             # Get next non-empty line
#             while line_idx < len(lines):
#                 line = lines[line_idx].strip()
#                 if line:  # if line is not empty
#                     break
#                 line_idx += 1
            
#             if line_idx >= len(lines):
#                 break
                
#             try:
#                 # Split the line and convert to numbers
#                 values = line.split()
#                 if not values:  # Skip if line is empty after splitting
#                     line_idx += 1
#                     continue
                    
#                 line_data = [float(x) for x in values]
                
#                 n_matches = int(line_data[0])
#                 current_u, current_v = line_data[4:6]
#                 matches_data = line_data[6:]  # Skip count, RGB, and current coords
                
#                 found_match = False  # Track if any match is found
                
#                 # Look through matches to find if either target image is present
#                 for i in range(0, len(matches_data), 3):
#                     if i + 2 >= len(matches_data):
#                         break
                        
#                     match_image_id = int(matches_data[i])
#                     match_u = matches_data[i + 1]
#                     match_v = matches_data[i + 2]
                    
#                     if match_image_id == image_id2:
#                         points1.append([current_u, current_v])
#                         points2.append([match_u, match_v])
#                         found_match = True
#                     elif match_image_id == image_id1:
#                         points2.append([current_u, current_v])
#                         points1.append([match_u, match_v])
#                         found_match = True
                
#                 # If no match found, increase the counter
#                 if not found_match:
#                     no_match_count += 1
            
#             except (ValueError, IndexError) as e:
#                 print(f"Warning: Error processing line {line_idx + 1}: {e}")
                
#             current_feature += 1
#             line_idx += 1

#     print(f"Total Features: {n_features}, Features with No Matches: {no_match_count}")
#     return np.array(points1), np.array(points2)

def read_matches_file(filename, image_id1, image_id2):
    """
    Parse matches.txt file to extract corresponding points between two specific images.
    Returns points and their corresponding line numbers.
    """
    points1 = []
    points2 = []
    line_numbers = []  # List to store line numbers where matches are found
    no_match_count = 0  # Initialize the variable
    
    with open(filename, 'r') as f:
        # Read and process all lines
        lines = f.readlines()
        
        # Find the first non-empty line
        n_features = 0
        for line in lines:
            line = line.strip()
            if line:  # if line is not empty
                if 'nFeatures' in line:
                    n_features = int(line.split(': ')[1])
                    break
        
        # Process each feature line
        current_feature = 0
        line_idx = 1  # Start from next line after nFeatures
        
        while current_feature < n_features and line_idx < len(lines):
            # Get next non-empty line
            while line_idx < len(lines):
                line = lines[line_idx].strip()
                if line:  # if line is not empty
                    break
                line_idx += 1
            
            if line_idx >= len(lines):
                break
                
            try:
                # Split the line and convert to numbers
                values = line.split()
                if not values:  # Skip if line is empty after splitting
                    line_idx += 1
                    continue
                    
                line_data = [float(x) for x in values]
                
                n_matches = int(line_data[0])
                current_u, current_v = line_data[4:6]
                matches_data = line_data[6:]  # Skip count, RGB, and current coords
                
                found_match = False  # Track if any match is found
                
                # Look through matches to find if either target image is present
                for i in range(0, len(matches_data), 3):
                    if i + 2 >= len(matches_data):
                        break
                        
                    match_image_id = int(matches_data[i])
                    match_u = matches_data[i + 1]
                    match_v = matches_data[i + 2]
                    
                    if match_image_id == image_id2:
                        points1.append([current_u, current_v])
                        points2.append([match_u, match_v])
                        line_numbers.append(line_idx + 1)  # +1 to convert zero-index to actual line number
                        found_match = True
                    elif match_image_id == image_id1:
                        points2.append([current_u, current_v])
                        points1.append([match_u, match_v])
                        line_numbers.append(line_idx + 1)  # +1 to convert zero-index to actual line number
                        found_match = True
                
                # If no match found, increase the counter
                if not found_match:
                    no_match_count += 1
            
            except (ValueError, IndexError) as e:
                print(f"Warning: Error processing line {line_idx + 1}: {e}")
                
            current_feature += 1
            line_idx += 1

    print(f"Total Features: {n_features}, Features with No Matches: {no_match_count}")
    return np.array(points1), np.array(points2), line_numbers


def EstimateFundamentalMatrix(points1, points2):

    # Convert points to homogeneous coordinates
    points1_h = np.hstack((points1, np.ones((points1.shape[0], 1))))
    points2_h = np.hstack((points2, np.ones((points2.shape[0], 1))))
    
    # Build the constraint matrix
    A = np.zeros((points1.shape[0], 9))
    for i in range(points1.shape[0]):
        x1, y1 = points1_h[i, :2]
        x2, y2 = points2_h[i, :2]
        A[i] = [x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1]
    
    # Solve for F using SVD
    _, _, V = np.linalg.svd(A)
    F = V[-1].reshape(3, 3)
    
    # Enforce rank 2 constraint
    U, S, V = np.linalg.svd(F)
    S[2] = 0
    F = U @ np.diag(S) @ V
    
    return F

# Just to check if the fundamental matrix is correct
def calculate_epipolar_error(F, points1, points2):
    """
    Calculate epipolar constraint error for the fundamental matrix.
    """
    points1_h = np.hstack((points1, np.ones((points1.shape[0], 1))))
    points2_h = np.hstack((points2, np.ones((points2.shape[0], 1))))
    
    # Calculate epipolar constraint x2.T * F * x1
    errors = []
    for i in range(len(points1)):
        error = abs(points2_h[i].dot(F.dot(points1_h[i])))
        errors.append(error)
    
    return np.mean(errors)

def GetInlierRANSANC(points1, points2, line_numbers, num_iterations=1000, threshold=0.01):

    best_F = None
    best_inliers = []
    best_inlier_count = 0
    best_inlier_lines = []
    
    if len(points1) < 8:
        print("Not enough points for RANSAC (need at least 8)")
        return None, [], []
    
    num_points = len(points1)
    
    for i in range(num_iterations):
        # 1. Randomly select 8 correspondences
        sample_indices = np.random.choice(num_points, 8, replace=False)
        sample_points1 = points1[sample_indices]
        sample_points2 = points2[sample_indices]
        
        # 2. Compute fundamental matrix from these samples
        F = EstimateFundamentalMatrix(sample_points1, sample_points2)
        
        # 3. Determine inliers based on epipolar constraint
        inliers = []
        inlier_lines = []
        
        # Convert points to homogeneous coordinates
        points1_h = np.hstack((points1, np.ones((num_points, 1))))
        points2_h = np.hstack((points2, np.ones((num_points, 1))))
        
        # Check each correspondence
        for j in range(num_points):
            # Calculate epipolar constraint error: |x2^T F x1|
            error = abs(points2_h[j].dot(F.dot(points1_h[j])))
            
            # If error is small enough, consider it an inlier
            if error < threshold:
                inliers.append(j)
                inlier_lines.append(line_numbers[j])
        
        # 4. Update best model if we found more inliers
        if len(inliers) > best_inlier_count:
            best_inlier_count = len(inliers)
            best_inliers = inliers
            best_F = F
            best_inlier_lines = inlier_lines
    
    print(f"RANSAC found {best_inlier_count} inliers out of {num_points} points")
    
    # Optionally, recompute F using all inliers for better accuracy
    if len(best_inliers) >= 8:
        inlier_points1 = points1[best_inliers]
        inlier_points2 = points2[best_inliers]
        best_F = EstimateFundamentalMatrix(inlier_points1, inlier_points2)
    
    return best_F, best_inliers, best_inlier_lines

def EssentialMatrixFromFundamentalMatrix(F, K):

    E = K.T @ F @ K
    U, _, Vt = np.linalg.svd(E)
    E = U @ np.diag([1, 1, 0]) @ Vt
    return E

def ExtractCameraPose(E):
    # SVD of Essential matrix
    U, S, Vt = np.linalg.svd(E)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    c1 = U[:, 2]
    c2 = -U[:, 2]

    if np.linalg.det(R1) < 0:
        R1 = -R1
        c1 = -c1
        c2 = -c2

    if np.linalg.det(R2) < 0:
        R2 = -R2
        c2 = -c2
        c1 = -c1

    return (R1, c1), (R2, c2), (R1, c2), (R2, c1)

def LinearTriangulation(K, R1, c1, R2, c2, x1, x2):
    # Make projection matrices
    P1 = K @ R1 @ np.hstack((np.eye(3), -c1.reshape(3,1)))
    P2 = K @ R2 @ np.hstack((np.eye(3), -c2.reshape(3,1)))

    # Convert image points to homogeneous coordinates
    x1_h = np.hstack((x1, np.ones((x1.shape[0], 1))))   # (N, 3)
    x2_h = np.hstack((x2, np.ones((x2.shape[0], 1))))   # (N, 3)    

    X = np.zeros((x1.shape[0], 3))
    for i in range(x1.shape[0]):
        A = np.vstack((x1_h[i, 0] * P1[2, :] - P1[0, :],
                       x1_h[i, 1] * P1[2, :] - P1[1, :],
                       x2_h[i, 0] * P2[2, :] - P2[0, :],
                       x2_h[i, 1] * P2[2, :] - P2[1, :]))
    
        _, _, Vt = np.linalg.svd(A)
        X[i] = Vt[-1, :3] / Vt[-1, 3]

    return X

def main():
    # Example usage with your matches.txt file
    matches_file = 'matching1.txt'
    image_id1 = 1  # Replace with your first image ID
    image_id2 = 2  # Replace with your second image ID
    
    # Parse matching points
    points1, points2, line_numbers = read_matches_file(matches_file, image_id1, image_id2)
    
    print(f"Found {len(points1)} potential matches")
    
    if len(points1) < 8:
        print(f"Not enough matches found between images {image_id1} and {image_id2}")
        print(f"Found {len(points1)} matches, need at least 8")
        return
    
    # Run RANSAC to find inliers and estimate F
    best_F, inliers, inlier_lines = GetInlierRANSANC(points1, points2, line_numbers)
    
    # Calculate error using inliers
    if len(inliers) >= 8:
        inlier_points1 = points1[inliers]
        inlier_points2 = points2[inliers]
        error = calculate_epipolar_error(best_F, inlier_points1, inlier_points2)
        
        print("\nFundamental Matrix:")
        print(best_F)
        print(f"\nNumber of inliers: {len(inliers)} out of {len(points1)} points")
        # print("Line numbers of inliers:", inlier_lines)
        print("\nAverage Epipolar Error for inliers:", error)
    else:
        print("Failed to find enough inliers using RANSAC")

    K = np.loadtxt('calibration.txt')
    E = EssentialMatrixFromFundamentalMatrix(best_F,K)
    

    poses = ExtractCameraPose(E)

    print("\nCamera Poses:")
    for i, pose in enumerate(poses):
        R, c = pose
        print(f"Pose {i+1}:\nR:\n{R}\nc:\n{c}")


    # Triangulate points
    X = LinearTriangulation(K, poses[0][0], poses[0][1], poses[1][0], poses[1][1], points1, points2)
    print("\nTriangulated 3D points:")
    print(X)
    print(X.shape)
    





if __name__ == "__main__":
    main()
