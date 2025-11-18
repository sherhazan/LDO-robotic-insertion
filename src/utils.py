import cv2
import numpy as np

def center_blue_object(rgb_image, depth_image):
    """
    Detects the largest blue object in the RGB image and centers both
    the RGB and Depth images around its centroid.
    """
    # Convert to HSV
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)

    # Define blue color range
    lower_blue = np.array([90, 100, 100])
    upper_blue = np.array([130, 255, 255])

    # Threshold the HSV image
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Find the largest contour
        max_contour = max(contours, key=cv2.contourArea)
        
        # Calculate centroid
        M = cv2.moments(max_contour)
        if M['m00'] == 0: return rgb_image, depth_image # Avoid division by zero
        
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        # Calculate shift
        rows, cols = rgb_image.shape[:2]
        offset_x = cols // 2 - cx
        offset_y = rows // 2 - cy

        # Apply shift (roll) to both images
        centered_rgb = np.roll(rgb_image, offset_x, axis=1)
        centered_rgb = np.roll(centered_rgb, offset_y, axis=0)

        centered_depth = np.roll(depth_image, offset_x, axis=1)
        centered_depth = np.roll(centered_depth, offset_y, axis=0)

        return centered_rgb, centered_depth
    else:
        # No object found, return original
        return rgb_image, depth_image