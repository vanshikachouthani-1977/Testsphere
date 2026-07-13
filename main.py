import os
import json
import base64
import asyncio
from typing import List
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from orchestrator import QAOrchestrator

# Initialize environment variables
from dotenv import load_dotenv
load_dotenv()

app = FastAPI(
    title="TestSphere QA Automation API",
    description="Python backend for comparing Figma mockups against website screenshots using multimodal AI and computer vision.",
    version="1.0.0"
)

# Ensure output directory exists for saving diff images
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Instantiate the central orchestrator
orchestrator = QAOrchestrator()

from fastapi.responses import FileResponse

@app.get("/", include_in_schema=False)
def root():
    index_path = os.path.join("static", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return RedirectResponse(url="/docs")

@app.post("/upload", summary="Upload Mockup & Screenshot for QA Audit")
async def upload_audit(
    mockup: UploadFile = File(..., description="The reference Figma mockup image file"),
    screenshot: UploadFile = File(..., description="The live website screenshot image file"),
    test_cases: str = Form("[]", description="JSON-serialized list of test cases, e.g. ['CTA button must be blue']")
):
    """
    Accepts Figma mockup and website screenshot files, runs vector and pixel comparison in parallel,
    runs Gemini Vision AI audit, maps findings to user test cases, and returns a unified report.
    """
    # 1. Parse test cases from Form string
    try:
        test_cases_list = json.loads(test_cases)
        if not isinstance(test_cases_list, list):
            raise ValueError()
    except Exception:
        # Fallback to splitting by comma if not valid JSON
        if test_cases:
            test_cases_list = [tc.strip() for tc in test_cases.split(",") if tc.strip()]
        else:
            test_cases_list = []

    # 2. Read image bytes
    try:
        mockup_bytes = await mockup.read()
        screenshot_bytes = await screenshot.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read uploaded image files: {str(e)}")

    # 3. Trigger orchestrator
    try:
        report = await orchestrator.run_audit(mockup_bytes, screenshot_bytes, test_cases_list)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"QA Audit execution failed: {str(e)}")

    # 4. Save and serve the difference heatmap image
    diff_filename = f"diff_{int(asyncio.get_event_loop().time())}.png"
    diff_filepath = os.path.join("static", diff_filename)
    
    try:
        # Get the last generated diff image from the orchestrator
        diff_bytes = getattr(orchestrator, "last_diff_image_bytes", b"")
        if diff_bytes:
            with open(diff_filepath, "wb") as f:
                f.write(diff_bytes)
            # Add URL to the report
            report["diff_image_url"] = f"/static/{diff_filename}"
            # Add base64 data URI to report for inline display convenience
            base64_img = base64.b64encode(diff_bytes).decode("utf-8")
            report["diff_image_base64"] = f"data:image/png;base64,{base64_img}"
        else:
            report["diff_image_url"] = None
            report["diff_image_base64"] = None
    except Exception as e:
        print(f"Warning: Failed to save diff heatmap image: {e}")
        report["diff_image_url"] = None
        report["diff_image_base64"] = None

    return report

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
