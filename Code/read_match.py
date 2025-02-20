import cv2
import numpy as np

def parse_matching_file(file_path):
    """Parses the matching file and returns feature matches."""
    with open(file_path, 'r') as f:
        lines = f.readlines()

    matches = []
    n_features = int(lines[0].split(":")[1].strip())  # Extract number of features
    
    for line in lines[1:]:
        values = list(map(float, line.split()))
        num_matches = int(values[0])  # Number of matches for this feature
        color = tuple(map(int, values[1:4]))  # RGB color of the feature
        u_curr, v_curr = values[4:6]  # Feature coordinates in current image
        
        feature_matches = []
        for i in range(num_matches - 1):  # -1 because first is self-match
            img_id = int(values[6 + 3 * i])  # Image ID where feature is matched
            u_match, v_match = values[7 + 3 * i], values[8 + 3 * i]  # Match coords
            feature_matches.append((img_id, (u_match, v_match)))
        
        matches.append(((u_curr, v_curr), color, feature_matches))
    
    return matches

def draw_matches(image1, image2, matches):
    """Draws feature matches between two images."""
    h1, w1, _ = image1.shape
    h2, w2, _ = image2.shape
    
    # Create a side-by-side image
    combined_img = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    combined_img[:h1, :w1] = image1
    combined_img[:h2, w1:] = image2
    green_color = (0, 255, 0) 
   
    
    for (u_curr, v_curr), color, feature_matches in matches:
        # Convert floating point coordinates to integer
        pt1 = (int(u_curr), int(v_curr))

        for img_id, (u_match, v_match) in feature_matches:
            # Assuming image2 corresponds to the matched image
            pt2 = (int(u_match) + w1, int(v_match))  # Adjust x-coord for concatenation
            
            # Draw feature points
            cv2.circle(combined_img, pt1, 3, green_color, -1)
            cv2.circle(combined_img, pt2, 3, green_color, -1)
            
            # Draw matching line
            cv2.line(combined_img, pt1, pt2, color, 1)

    return combined_img

# Load images
image1 = cv2.imread(r"C:\Users\farha\OneDrive\Desktop\P2Data\P2Data\4.png")  # Change to your image file
image2 = cv2.imread(r"C:\Users\farha\OneDrive\Desktop\P2Data\P2Data\5.png")  # Change to your image file

# Parse matches
matches = parse_matching_file(r"C:\Users\farha\OneDrive\Desktop\P2Data\P2Data\matching4.txt")  # Change to your file

# Draw matches
result_img = draw_matches(image1, image2, matches)

# Show image
cv2.imshow("Feature Matches", result_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite(r"C:\Users\farha\OneDrive\Desktop\P2Data\P2Data\matches.png", result_img)  # Save image
