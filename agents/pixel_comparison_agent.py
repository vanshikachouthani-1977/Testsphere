import io
import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

class PixelComparisonAgent:
    def __init__(self):
        pass

    def compare(self, mockup_img: Image.Image, screenshot_img: Image.Image) -> tuple[float, bytes]:
        """
        Compares two images using the Structural Similarity Index (SSIM).
        Generates a diff heatmap image highlighting differences in red.
        """
        # Convert PIL Images to NumPy arrays
        mockup_np = np.array(mockup_img)
        screenshot_np = np.array(screenshot_img)

        # Convert to grayscale for SSIM calculation
        mockup_gray = cv2.cvtColor(mockup_np, cv2.COLOR_RGB2GRAY)
        screenshot_gray = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2GRAY)

        # Calculate SSIM
        # full=True returns the full structural similarity image
        score, diff_map = ssim(mockup_gray, screenshot_gray, full=True)
        
        # Convert diff_map from float range [-1, 1] to uint8 [0, 255]
        # Lower values mean higher difference
        diff_uint8 = ((1.0 - diff_map) * 127.5).astype(np.uint8)

        # Apply threshold to capture significant differences
        _, thresh = cv2.threshold(diff_uint8, 30, 255, cv2.THRESH_BINARY)

        # Create a colored heatmap overlay (red)
        # We start with the original mockup in BGR format
        heatmap_bgr = cv2.cvtColor(mockup_np, cv2.COLOR_RGB2BGR)

        # Generate a red mask
        red_mask = np.zeros_like(heatmap_bgr)
        red_mask[:, :] = [0, 0, 255]  # Red color in BGR

        # Combine original image with red mask where difference exceeds threshold
        # Using a 0.5 opacity overlay
        alpha = 0.5
        mask = thresh > 0
        heatmap_bgr[mask] = cv2.addWeighted(heatmap_bgr, 1 - alpha, red_mask, alpha, 0)[mask]

        # Convert back to RGB for Pillow saving
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

        # Save the heatmap image to bytes
        diff_pil = Image.fromarray(heatmap_rgb)
        buf = io.BytesIO()
        diff_pil.save(buf, format="PNG")
        diff_bytes = buf.getvalue()

        # Score is between -1 and 1. Normalize to 0 to 1 range
        normalized_score = max(0.0, float((score + 1.0) / 2.0))

        return normalized_score, diff_bytes
