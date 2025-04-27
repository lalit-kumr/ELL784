import cv2
import os
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# ---------------------------------------------
# 1. Extract Frames from Video
# ---------------------------------------------
def extract_frames(video_path, resize_dim=(128, 128)):
    """
    Extract frames from a video and resize them for efficient processing.
    
    Args:
    video_path (str): Path to the input video file.
    resize_dim (tuple): Target frame dimensions (width, height) for resizing.
    
    Returns:
    list: A list of frames (numpy arrays).
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    success, frame = cap.read()
    
    while success:
        # Resize frame to reduce computation overhead
        frame_resized = cv2.resize(frame, resize_dim)
        frames.append(frame_resized)
        success, frame = cap.read()
    
    cap.release()
    print(f"Extracted {len(frames)} frames.")
    return frames

# ---------------------------------------------
# 2. Apply PCA for Feature Extraction
# ---------------------------------------------
def apply_pca(frames, n_components=10):
    """
    Apply PCA to reduce frame dimensions and identify key components.
    
    Args:
    frames (list): List of video frames (numpy arrays).
    n_components (int): Number of principal components to retain.
    
    Returns:
    tuple: Transformed PCA features and the PCA model.
    """
    # Flatten each frame into a 1D array and stack into a matrix
    data = np.array([frame.flatten() for frame in frames])
    
    # Apply PCA to the frame data
    pca = PCA(n_components=n_components)
    pca_features = pca.fit_transform(data)
    
    return pca_features, pca

# ---------------------------------------------
# 3. Select Keyframes Based on PCA Distance
# ---------------------------------------------
def select_keyframes(pca_features, frames, num_keyframes=5):
    """
    Select keyframes based on their distance from the mean PCA component.
    
    Args:
    pca_features (ndarray): PCA features of frames.
    frames (list): Original frames corresponding to PCA features.
    num_keyframes (int): Number of keyframes to select.
    
    Returns:
    list: Selected keyframes.
    """
    # Compute the euclidean distance of each frame from the mean PCA vector
    mean_vector = np.mean(pca_features, axis=0)
    distances = np.linalg.norm(pca_features - mean_vector, axis=1)
    
    # Select indices with the largest distances
    keyframe_indices = distances.argsort()[-num_keyframes:]
    
    # Sort indices to maintain temporal order
    keyframes = [frames[i] for i in sorted(keyframe_indices)]
    return keyframes

# ---------------------------------------------
# 4. Save Summarized Video
# ---------------------------------------------
def save_summarized_video(keyframes, output_path="summarized_video.avi", fps=5):
    """
    Save keyframes as a summarized video.
    
    Args:
    keyframes (list): List of selected keyframes.
    output_path (str): Path to save the output video.
    fps (int): Frames per second for the summarized video.
    """
    height, width, layers = keyframes[0].shape
    size = (width, height)
    
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, size)
    for frame in keyframes:
        out.write(frame)
    out.release()
    print(f"Summarized video saved at {output_path}")

# ---------------------------------------------
# 5. Elbow Method for Optimal Frame Selection
# ---------------------------------------------
def plot_elbow_curve(pca_features):
    """
    Plot the elbow curve to determine the optimal number of keyframes.
    
    Args:
    pca_features (ndarray): PCA features of frames.
    """
    distortions = []
    N = 999
    for k in range(1, N):
        # Compute sum of distances for k keyframes
        distances = np.linalg.norm(pca_features - np.mean(pca_features[:k], axis=0), axis=1)
        distortions.append(np.sum(distances))
    
    # Plot the elbow curve
    plt.plot(range(1, N), distortions)
    plt.xlabel("Number of Keyframes")
    plt.ylabel("Distortion")
    plt.title("Elbow Method for Optimal Keyframe Selection")
    plt.show()

# ---------------------------------------------
# Main Execution
# ---------------------------------------------
if __name__ == "__main__":
    # Replace with the path to your video file
    video_path = "video.mpeg"
    
    # Step 1: Extract frames
    frames = extract_frames(video_path)
    
    # Step 2: Apply PCA to reduce dimensions
    pca_features, pca_model = apply_pca(frames, n_components=5)
    
    # Step 3: Select keyframes and save summarized video
    keyframes = select_keyframes(pca_features, frames, num_keyframes=10)
    save_summarized_video(keyframes)
    
    # Step 4: Experiment with different keyframe counts
    for num_frames in [5, 25, 50, 100, 200, 400]:
        selected_frames = select_keyframes(pca_features, frames, num_keyframes=num_frames)
        save_summarized_video(selected_frames, output_path=f"summarized_{num_frames}_frames.avi")
    
    # Step 5: Plot elbow curve to select optimal number of keyframes
    plot_elbow_curve(pca_features)
