# import cv2
# import numpy as np
# from scipy.stats import norm

# class CustomMOG2:
#     def __init__(self, n_gaussians=5, learning_rate=0.01, initial_variance=36.0, 
#                  min_variance=4.0, background_ratio=0.9, min_weight=0.05):
#         """
#         Custom implementation of MOG2 background subtractor.
#         """
#         self.n_gaussians = n_gaussians
#         self.learning_rate = learning_rate
#         self.initial_variance = initial_variance
#         self.min_variance = min_variance
#         self.background_ratio = background_ratio
#         self.min_weight = min_weight
        
#         self.means = None
#         self.variances = None
#         self.weights = None
#         self.background_mask = None
        
#     def _initialize(self, frame):
#         """Initialize model components for the first frame."""
#         height, width = frame.shape
        
#         self.means = np.zeros((height, width, self.n_gaussians))
#         self.variances = np.full((height, width, self.n_gaussians), 
#                                self.initial_variance)
#         self.weights = np.full((height, width, self.n_gaussians), 
#                              1.0 / self.n_gaussians)
        
#         self.means[..., 0] = frame
        
#     def _match_gaussians(self, frame):
#         """Find matching Gaussians for each pixel value."""
#         diff = frame[..., np.newaxis] - self.means
#         mahalanobis_dist = diff * diff / self.variances
#         matches = mahalanobis_dist < 6.25  # (2.5 * 2.5)
#         return matches
    
#     def _update_gaussians(self, frame, matches):
#         """Update Gaussian parameters based on matches."""
#         lr = self.learning_rate
        
#         weight_update = np.where(matches, lr, -lr)
#         self.weights += weight_update
#         self.weights = np.clip(self.weights, self.min_weight, 1.0)
#         self.weights /= np.sum(self.weights, axis=2)[..., np.newaxis]
        
#         match_indices = np.argmax(matches, axis=2)
#         for i in range(self.n_gaussians):
#             mask = match_indices == i
#             if np.any(mask):
#                 diff = frame[mask] - self.means[mask, i]
#                 self.means[mask, i] += lr * diff
#                 diff_sq = diff * diff
#                 var_update = lr * (diff_sq - self.variances[mask, i])
#                 self.variances[mask, i] += var_update
#                 self.variances[mask, i] = np.maximum(self.variances[mask, i], 
#                                                    self.min_variance)
    
#     def _create_new_gaussians(self, frame, matches):
#         """Create new Gaussians for unmatched pixels."""
#         unmatched = ~np.any(matches, axis=2)
#         if np.any(unmatched):
#             min_weight_idx = np.argmin(self.weights[unmatched], axis=1)
            
#             self.means[unmatched, min_weight_idx] = frame[unmatched]
#             self.variances[unmatched, min_weight_idx] = self.initial_variance
#             self.weights[unmatched, min_weight_idx] = self.min_weight
#             self.weights[unmatched] /= np.sum(self.weights[unmatched], 
#                                             axis=1)[:, np.newaxis]
    
#     def _determine_background(self):
#         """Determine which Gaussians represent background."""
#         weight_var_ratio = self.weights / np.sqrt(self.variances)
#         sorted_indices = np.argsort(weight_var_ratio, axis=2)[..., ::-1]
        
#         sorted_weights = np.take_along_axis(self.weights, 
#                                           sorted_indices, axis=2)
#         cumsum = np.cumsum(sorted_weights, axis=2)
        
#         background_indices = cumsum <= self.background_ratio
#         return background_indices, sorted_indices
    
#     def apply(self, frame):
#         """Apply MOG2 background subtraction to a frame."""
#         if self.means is None:
#             self._initialize(frame)
#             return np.zeros_like(frame, dtype=np.uint8)
        
#         matches = self._match_gaussians(frame)
#         self._update_gaussians(frame, matches)
#         self._create_new_gaussians(frame, matches)
        
#         background_indices, sorted_indices = self._determine_background()
        
#         matched_gaussians = np.argmax(matches, axis=2)
#         background_gaussian_indices = np.take_along_axis(
#             matched_gaussians[..., np.newaxis], 
#             sorted_indices, 
#             axis=2
#         )
#         is_background = np.any(
#             np.logical_and(background_indices, 
#                           background_gaussian_indices == sorted_indices),
#             axis=2
#         )
        
#         return np.where(is_background, 0, 255).astype(np.uint8)
    
#     def getBackgroundImage(self):
#         """Get the current background image."""
#         if self.means is None:
#             return None
            
#         weight_var_ratio = self.weights / np.sqrt(self.variances)
#         best_gaussian = np.argmax(weight_var_ratio, axis=2)
#         height, width = self.means.shape[:2]
#         background = np.zeros((height, width))
        
#         for i in range(height):
#             for j in range(width):
#                 background[i, j] = self.means[i, j, best_gaussian[i, j]]
                
#         return background.astype(np.uint8)

# def process_video(video_path, n_gaussians=5, learning_rate=0.01):
#     """
#     Process video using custom MOG2 background subtractor.
#     """
#     # Create custom background subtractor
#     background_subtractor = CustomMOG2(
#         n_gaussians=n_gaussians,
#         learning_rate=learning_rate
#     )
    
#     # Open video file
#     cap = cv2.VideoCapture('video.mpeg')
#     if not cap.isOpened():
#         raise ValueError("Error opening video.mpeg")
    
#     # Get video properties
#     frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = int(cap.get(cv2.CAP_PROP_FPS))
    
#     # Create video writers
#     foreground_out = cv2.VideoWriter(
#         'foreground.mp4',
#         cv2.VideoWriter_fourcc(*'mp4v'),
#         fps,
#         (frame_width, frame_height)
#     )
    
#     background_out = cv2.VideoWriter(
#         'background.mp4',
#         cv2.VideoWriter_fourcc(*'mp4v'),
#         fps,
#         (frame_width, frame_height)
#     )
    
#     try:
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break
            
#             # Convert frame to grayscale
#             gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
#             # Apply background subtraction
#             fg_mask = background_subtractor.apply(gray_frame)
            
#             # Get background image
#             bg_image = background_subtractor.getBackgroundImage()
            
#             if bg_image is None:
#                 continue
            
#             # Convert outputs to 3-channel for saving
#             fg_mask_3d = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
#             bg_image_3d = cv2.cvtColor(bg_image, cv2.COLOR_GRAY2BGR)
            
#             # Create foreground by applying mask
#             gray_frame_3d = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
#             foreground = cv2.bitwise_and(gray_frame_3d, fg_mask_3d)
            
#             # Write frames
#             foreground_out.write(foreground)
#             background_out.write(bg_image_3d)
            
#             # Display results
#             cv2.imshow('Original', gray_frame)
#             cv2.imshow('Background', bg_image)
#             cv2.imshow('Foreground', foreground)
            
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
                
#     finally:
#         cap.release()
#         foreground_out.release()
#         background_out.release()
#         cv2.destroyAllWindows()

# def main():
#     try:
#         process_video(
#             video_path='video.mpeg',
#             n_gaussians=5,      # Number of Gaussian components per pixel
#             learning_rate=0.01  # Learning rate for model updates
#         )
#     except Exception as e:
#         print(f"Error processing video: {e}")

# if __name__ == "__main__":
#     main()



import cv2
import numpy as np

def process_video(video_path, history=500, var_threshold=16, detect_shadows=True):
    """
    Apply Stauffer-Grimson background subtraction to a grayscale version of the video
    and save background and foreground in MP4 format.
    
    Parameters:
    video_path (str): Path to the video file
    history (int): Length of history buffer for background model
    var_threshold (float): Threshold for foreground detection
    detect_shadows (bool): Whether to detect and mark shadows
    """
    # Create background subtractor
    background_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=history,
        varThreshold=var_threshold,
        detectShadows=detect_shadows
    )
    
    # Open video file
    cap = cv2.VideoCapture('video.mpeg')
    if not cap.isOpened():
        raise ValueError("Error opening video.mpeg")
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Create video writers for output using MP4/H264 codec
    foreground_out = cv2.VideoWriter(
        'foreground.mp4',
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (frame_width, frame_height)
    )
    
    background_out = cv2.VideoWriter(
        'background.mp4',
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (frame_width, frame_height)
    )
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert frame to grayscale
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply background subtraction on grayscale frame
            fg_mask = background_subtractor.apply(gray_frame)
            
            # Get the background image (will be grayscale)
            bg_image = background_subtractor.getBackgroundImage()
            
            if bg_image is None:
                continue  # Skip if background model isn't ready yet
            
            # Convert mask and background to 3-channel for saving
            fg_mask_3d = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
            bg_image_3d = cv2.cvtColor(bg_image, cv2.COLOR_GRAY2BGR)
            
            # Create foreground by applying mask to original grayscale frame
            gray_frame_3d = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
            foreground = cv2.bitwise_and(gray_frame_3d, fg_mask_3d)
            
            # Write frames to respective output videos
            foreground_out.write(foreground)
            background_out.write(bg_image_3d)
            
            # Display results (comment out for faster processing)
            cv2.imshow('Grayscale Input', gray_frame)
            cv2.imshow('Background', bg_image)
            cv2.imshow('Foreground', foreground)
            
            # Break loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        # Clean up
        cap.release()
        foreground_out.release()
        background_out.release()
        cv2.destroyAllWindows()

def main():
    try:
        process_video(
            video_path='video.mpeg',
            history=500,        # Number of frames to build background model
            var_threshold=16,   # Threshold for foreground detection
            detect_shadows=True # Enable shadow detection
        )
    except Exception as e:
        print(f"Error processing video: {e}")

if __name__ == "__main__":
    main()