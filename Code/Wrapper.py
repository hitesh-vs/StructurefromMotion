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

def EssentialMatrixFromFundamentalMatrix(F, K):
    E = K.T @ F @ K
    # Normalize the Essential matrix
    U, S, Vt = np.linalg.svd(E)
    # Force singular values to [1,1,0]
    S = np.array([1, 1, 0])
    E = U @ np.diag(S) @ Vt
    return E




# Complete example usage
def main():
    # Example usage with your matches.txt file
    matches_file = r'matching1.txt'
    image_id1 = 1  # Replace with your first image ID
    image_id2 = 2  # Replace with your second image ID
    
    # Parse matching points
    points1, points2 , check = read_matches_file(matches_file, image_id1, image_id2)
    
    print(check) 
    if len(points1) < 8:
        print(f"Not enough matches found between images {image_id1} and {image_id2}")
        print(f"Found {len(points1)} matches, need at least 8")
        return
    
    # Compute fundamental matrix
    F = EstimateFundamentalMatrix(points1, points2)
    
    # Calculate error
    error = calculate_epipolar_error(F, points1, points2)

    # Load K matrix
    K = np.loadtxt('calibration.txt')
    E = EssentialMatrixFromFundamentalMatrix(F,K)

    print("K matrix:\n",K)
    print("E\n",E)
    
    print("Fundamental Matrix:")
    print(F)
    print("\nAverage Epipolar Error:", error)
    
    return F, points1, points2

if __name__ == "__main__":
    main()
