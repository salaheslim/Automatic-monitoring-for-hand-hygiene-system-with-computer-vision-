import numpy as np

def wrist_anchor_normalise(features):
    """
    Normalise 126 hand landmark features relative to wrist position.
    
    Input: 126 features = [left_hand(63), right_hand(63)]
    Each hand = 21 landmarks x 3 coordinates (x,y,z)
    Landmark 0 = wrist
    
    Output: 126 normalised features
    """
    normalised = features.copy()
    
    # Process left hand (features 0-62)
    # Wrist is landmark 0: features[0]=x, features[1]=y, features[2]=z
    left_wrist_x = features[0]
    left_wrist_y = features[1]
    left_wrist_z = features[2]
    
    # Only normalise if hand is detected (non-zero wrist)
    if left_wrist_x != 0 or left_wrist_y != 0:
        for i in range(21):
            normalised[i*3]   = features[i*3]   - left_wrist_x
            normalised[i*3+1] = features[i*3+1] - left_wrist_y
            normalised[i*3+2] = features[i*3+2] - left_wrist_z

    # Process right hand (features 63-125)
    right_wrist_x = features[63]
    right_wrist_y = features[64]
    right_wrist_z = features[65]

    if right_wrist_x != 0 or right_wrist_y != 0:
        for i in range(21):
            normalised[63+i*3]   = features[63+i*3]   - right_wrist_x
            normalised[63+i*3+1] = features[63+i*3+1] - right_wrist_y
            normalised[63+i*3+2] = features[63+i*3+2] - right_wrist_z

    return normalised

# Test the function
if __name__ == "__main__":
    # Create dummy features with wrist at (0.5, 0.5, 0.0)
    test = np.zeros(126)
    test[0] = 0.5   # left wrist x
    test[1] = 0.5   # left wrist y
    test[2] = 0.0   # left wrist z
    test[3] = 0.6   # index finger x
    test[4] = 0.4   # index finger y

    result = wrist_anchor_normalise(test)
    print("Original wrist position:", test[0], test[1])
    print("Normalised wrist position:", result[0], result[1])
    print("Original index x:", test[3])
    print("Normalised index x:", result[3])
    print("Expected normalised index x: 0.1 (0.6 - 0.5)")
    print("\nNormalisation working:", abs(result[3] - 0.1) < 0.001)
