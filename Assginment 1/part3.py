import cv2
import numpy as np
import os
import re

class StaufferGrimsonBGS:
    def __init__(self, video_path, output_bg, output_fg, history=500, var_threshold=16, alpha=0.01, num_gaussians=5):
        self.video_path = video_path
        self.output_bg = output_bg
        self.output_fg = output_fg
        self.history = history
        self.var_threshold = var_threshold
        self.alpha = alpha
        self.num_gaussians = num_gaussians
        self.background_model = None
    
    def initialize_model(self, frame_shape):
        height, width = frame_shape
        self.background_model = np.zeros((height, width, self.num_gaussians, 3))  # mean, variance, weight
        self.background_model[..., 1] = 15  # Initial variance
        self.background_model[..., 2] = 1 / self.num_gaussians  # Initial weights
    
    def update_model(self, frame):
        height, width = frame.shape
        if self.background_model is None:
            self.initialize_model((height, width))
        
        fg_mask = np.ones((height, width), dtype=np.uint8) * 255  # Assume all pixels are foreground
        best_match = np.full((height, width), -1, dtype=int)  # Store best matching Gaussian index
        
        for i in range(self.num_gaussians):
            mean = self.background_model[..., i, 0]
            variance = self.background_model[..., i, 1]
            weight = self.background_model[..., i, 2]
            
            diff = np.abs(frame - mean)
            match = diff <= self.var_threshold * np.sqrt(variance)
            
            # Update matched distributions
            weight[match] = (1 - self.alpha) * weight[match] + self.alpha
            mean[match] = (1 - self.alpha) * mean[match] + self.alpha * frame[match]
            variance[match] = (1 - self.alpha) * variance[match] + self.alpha * (frame[match] - mean[match]) ** 2
            
            # Track the best matching Gaussian
            best_match[match] = i
        
        # Mark pixels as background where a match was found
        fg_mask[best_match != -1] = 0
        
        # Normalize weights
        total_weight = np.sum(self.background_model[..., 2], axis=2, keepdims=True)
        self.background_model[..., 2] /= total_weight
        
        return fg_mask
    
    def process_video(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Error: Unable to open video file {self.video_path}.")
            return
        
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_bg = cv2.VideoWriter(self.output_bg, fourcc, fps, (frame_width, frame_height), isColor=False)
        out_fg = cv2.VideoWriter(self.output_fg, fourcc, fps, (frame_width, frame_height), isColor=False)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            fg_mask = self.update_model(gray_frame)
            bg_image = np.sum(self.background_model[..., :, 0] * self.background_model[..., :, 2], axis=2).astype(np.uint8)
            
            out_fg.write(fg_mask)
            out_bg.write(bg_image)
            
        cap.release()
        out_fg.release()
        out_bg.release()
        cv2.destroyAllWindows()
        print(f"Processing complete for {self.video_path}. Outputs saved.")

# Find all video files matching 'summary_x_frames.avi'
video_files = [f for f in os.listdir('.') if re.match(r'summary_\d+_frames\.avi', f)]

for video_file in video_files:
    output_bg = f"bg_{video_file.split('.')[0]}.mp4"
    output_fg = f"fg_{video_file.split('.')[0]}.mp4"
    print(f"Processing {video_file}...")
    bgs = StaufferGrimsonBGS(video_file, output_bg, output_fg)
    bgs.process_video()
