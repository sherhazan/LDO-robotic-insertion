import cv2
import numpy as np

def random_crop_resize(rgb_img, depth_img, num_crops=3, output_size=(240, 240)):
    """
    Generates random crops from the center area of the image and resizes them.
    Returns a list of tuples: [(cropped_rgb, cropped_depth), ...]
    """
    h, w, _ = rgb_img.shape
    generated_images = []

    for _ in range(num_crops):
        # Randomize crop dimensions and shifts
        crop_base_size = np.random.randint(300, 500)
        
        # Random offsets for width and height variability (aspect ratio jitter)
        offset_h = np.random.randint(-50, 50) # corresponds to rand_1 in original code
        offset_w = np.random.randint(-50, 50) # corresponds to rand_2 in original code
        
        # Calculate crop dimensions based on original logic
        # Height is determined by crop_base_size + offset_h
        # Width is determined by crop_base_size + offset_w
        
        # Define boundaries (simulating the original logic)
        # Original: rgb_img[(h-crop_1)//2:(h+crop_1)//2, (w-crop_2)//2-rand_1:(w+crop_2)//2+rand_2]
        
        crop_h = crop_base_size + offset_h
        crop_w = crop_base_size + offset_w
        
        # Center calculation
        start_y = (h - crop_h) // 2
        end_y = (h + crop_h) // 2
        
        # The original code had a specific shift in X axis:
        # start_x = (w - crop_2)//2 - rand_1
        # end_x = (w + crop_2)//2 + rand_2
        # We replicate this specific jitter:
        start_x = (w - crop_w) // 2 - offset_h # relying on offset_h as rand_1
        end_x = (w + crop_w) // 2 + offset_w   # relying on offset_w as rand_2

        # Safety bounds (clip to image size) to prevent crashes
        start_y = max(0, start_y)
        end_y = min(h, end_y)
        start_x = max(0, start_x)
        end_x = min(w, end_x)

        # Perform Crop
        rgb_crop = rgb_img[start_y:end_y, start_x:end_x]
        depth_crop = depth_img[start_y:end_y, start_x:end_x]

        # Resize
        try:
            rgb_resized = cv2.resize(rgb_crop, output_size)
            depth_resized = cv2.resize(depth_crop, output_size)
            generated_images.append((rgb_resized, depth_resized))
        except Exception as e:
            print(f"Skipping bad crop: {start_x}:{end_x}, {start_y}:{end_y}. Error: {e}")

    return generated_images