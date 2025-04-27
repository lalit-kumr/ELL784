import os
import shutil
import cv2
import numpy as np
from sklearn.cluster import KMeans
import math

# Configuration Parameters
video_file = "./video.mpeg"
frames_dir = "./frames/"
frame_interval = 1  # Capture every Nth frame
num_gaussians = 3  # Number of Gaussians in GMM
learning_rate = 0.03  # Alpha value
threshold = 0.9  # Background threshold
rho = 0.1  # Update rate


def initialize_gmm(frame, total_pixels):
    """Initialize Gaussian Mixture Model using K-Means clustering."""
    frame = frame.reshape((total_pixels, 1))
    kmeans = KMeans(n_clusters=num_gaussians, n_init=10)
    kmeans.fit(frame)
    print("K-Means converged after", kmeans.n_iter_, "iterations")
    
    variances = np.full(num_gaussians, kmeans.inertia_ / total_pixels)
    means = kmeans.cluster_centers_.flatten()
    labels = kmeans.labels_
    
    r_matrix = np.zeros((total_pixels, num_gaussians))
    for j in range(num_gaussians):
        r_matrix[:, j] = (labels == j).astype(int)
    
    omega = (1 / num_gaussians) * (1 - learning_rate) + learning_rate * r_matrix
    return variances, means, omega


def process_video_frames(video_capture, total_pixels, video_width, video_height):
    """Extract and process frames from the video."""
    if os.path.exists(frames_dir):
        shutil.rmtree(frames_dir)
    os.makedirs(frames_dir)

    frame_id = 0
    success, frame = video_capture.read()
    if not success or frame is None:
        print("Error: Could not read the first frame. Check the video file.")
        return 0

    if frame.ndim == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    variances, means, omega = initialize_gmm(frame, total_pixels)
    background = np.zeros(total_pixels, dtype=np.uint8)
    foreground = np.zeros(total_pixels, dtype=np.uint8)

    while video_capture.isOpened():
        success, frame = video_capture.read()
        if not success or frame is None:
            break

        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        frame_id += 1
        if frame_id % frame_interval == 0:
            background, foreground = update_gmm(frame, variances, omega, means, background, foreground, total_pixels)
            print(f"Processing frame {frame_id}")
            
            background_img = background.reshape(video_height, video_width)
            foreground_img = foreground.reshape(video_height, video_width)
            
            cv2.imwrite(f"{frames_dir}/foreground_{str(frame_id).zfill(4)}.jpg", foreground_img)
            cv2.imwrite(f"{frames_dir}/background_{str(frame_id).zfill(4)}.jpg", background_img)

    return frame_id


def update_gmm(frame, variances, omega, means, background, foreground, total_pixels):
    """Update GMM for each pixel in the frame."""
    frame = frame.reshape((total_pixels,))
    background = background.reshape((total_pixels,))
    foreground = foreground.reshape((total_pixels,))
    ratio = np.zeros(num_gaussians)

    for pixel in range(total_pixels):
        matched = False
        sum_omega = 0
        for j in range(num_gaussians):
            if abs(frame[pixel] - means[j]) < 2.5 * (variances[j] ** 0.5):
                means[j] = (1 - rho) * means[j] + rho * frame[pixel]
                variances[j] = (1 - rho) * variances[j] + rho * ((frame[pixel] - means[j]) ** 2)
                omega[j] = (1 - learning_rate) * omega[j] + learning_rate
                matched = True
            else:
                omega[j] *= (1 - learning_rate)
            sum_omega += omega[j]

        omega /= sum_omega
        ratio = omega / variances
        sorted_indices = np.argsort(-ratio)
        means, variances, omega = means[sorted_indices], variances[sorted_indices], omega[sorted_indices]

        if not matched:
            means[-1] = frame[pixel]
            variances[-1] = 9999

        cumulative_weight = np.cumsum(omega)
        background_threshold = np.where(cumulative_weight > threshold)[0][0]

        if not matched or abs(frame[pixel] - means[background_threshold]) > 2.5 * (variances[background_threshold] ** 0.5):
            foreground[pixel] = frame[pixel]
            background[pixel] = means[background_threshold]
        else:
            foreground[pixel] = 255
            background[pixel] = frame[pixel]

    return background, foreground


def compile_video(frame_count, video_width, video_height, file_name):
    """Compile processed frames back into a video."""
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_video = cv2.VideoWriter(f"{file_name}.avi", fourcc, 30.0, (video_width, video_height))
    
    for i in range(frame_count - 1):
        frame_path = f"{frames_dir}/{file_name}_{str(i).zfill(4)}.jpg"
        frame = cv2.imread(frame_path)
        if frame is not None:
            output_video.write(frame)
    output_video.release()


if __name__ == "__main__":
    video_cap = cv2.VideoCapture(video_file)
    
    if not video_cap.isOpened():
        print("Error: Could not open video file.")
    else:
        width = round(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = round(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_pixels = width * height
        total_frames = process_video_frames(video_cap, total_pixels, width, height)
        
        compile_video(total_frames, width, height, "foreground")
        compile_video(total_frames, width, height, "background")
        
        video_cap.release()
        cv2.destroyAllWindows()
        print("Processing complete.")
