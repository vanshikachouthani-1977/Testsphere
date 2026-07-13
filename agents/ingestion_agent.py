import io
from PIL import Image

class IngestionAgent:
    def __init__(self):
        pass

    def process_images(self, mockup_bytes: bytes, screenshot_bytes: bytes) -> tuple[Image.Image, Image.Image]:
        """
        Loads, validates, and normalizes the uploaded mockup and screenshot images.
        Resizes the screenshot to match the exact dimensions of the mockup.
        """
        try:
            mockup_img = Image.open(io.BytesIO(mockup_bytes))
            # Verify the image is readable
            mockup_img.verify()
            # Re-open after verify() since verify() closes/invalidates the file pointer
            mockup_img = Image.open(io.BytesIO(mockup_bytes))
            mockup_img = mockup_img.convert("RGB")
        except Exception as e:
            raise ValueError(f"Invalid or corrupted mockup image: {str(e)}")

        try:
            screenshot_img = Image.open(io.BytesIO(screenshot_bytes))
            screenshot_img.verify()
            screenshot_img = Image.open(io.BytesIO(screenshot_bytes))
            screenshot_img = screenshot_img.convert("RGB")
        except Exception as e:
            raise ValueError(f"Invalid or corrupted screenshot image: {str(e)}")

        # Normalize the screenshot dimensions to match the mockup exactly
        mockup_width, mockup_height = mockup_img.size
        
        # We resize the screenshot using high-quality LANCZOS resizing
        screenshot_resized = screenshot_img.resize((mockup_width, mockup_height), Image.Resampling.LANCZOS)

        return mockup_img, screenshot_resized
