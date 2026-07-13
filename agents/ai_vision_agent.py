import os
import json
import io
import base64
import asyncio
import httpx
from PIL import Image
import google.generativeai as genai

class AIVisionAgent:
    def __init__(self):
        # Load environment variables if not loaded
        from dotenv import load_dotenv
        load_dotenv()
        
        self.gemini_key = os.getenv("GEMINI_API_KEY")
        self.groq_key = os.getenv("GROQ_API_KEY")
        
        if not self.gemini_key and not self.groq_key:
            raise ValueError("Neither GEMINI_API_KEY nor GROQ_API_KEY is set in environment or .env file.")
        
        if self.gemini_key:
            genai.configure(api_key=self.gemini_key)
            self.model = genai.GenerativeModel("gemini-1.5-flash")
            print("AI Vision Agent initialized using Gemini.")
        else:
            print("AI Vision Agent initialized using Groq Llama 3.2 Vision.")

    async def audit(self, mockup_img: Image.Image, screenshot_img: Image.Image) -> list:
        """
        Runs visual comparison on Mockup (Image 1) and Screenshot (Image 2).
        Returns a list of structured findings JSON objects.
        """
        if self.gemini_key:
            return await self._audit_gemini(mockup_img, screenshot_img)
        else:
            return await self._audit_groq(mockup_img, screenshot_img)

    async def _audit_gemini(self, mockup_img: Image.Image, screenshot_img: Image.Image) -> list:
        loop = asyncio.get_running_loop()

        system_prompt = (
            "You are a Senior QA Automation Engineer performing a category-by-category visual audit "
            "comparing a Figma mockup (Image 1) against a live website screenshot (Image 2).\n\n"
            "Analyze visual differences in the following categories:\n"
            "- layout (grid alignment, responsiveness structural columns)\n"
            "- component position (offsets, alignment, ordering)\n"
            "- sizing (height, width, button dimensions)\n"
            "- color (background, borders, branding color deviations)\n"
            "- typography (fonts, weights, sizes, styles)\n"
            "- spacing (margins, padding, row/column gaps)\n"
            "- missing/extra elements (headers, footers, buttons, icons)\n"
            "- styling details (border radius, box shadows, gradients)\n"
            "- images (missing graphic content, aspect ratio distortion)\n"
            "- responsive issues (text wrapping, overflow elements)\n\n"
            "You must return a JSON object with a single key \"findings\" which is a list of findings.\n"
            "Each finding in the list MUST be an object with the following fields:\n"
            "- \"category\": string matching one of the categories above\n"
            "- \"severity\": string value of exactly: \"Critical\", \"High\", \"Medium\", or \"Low\"\n"
            "- \"finding\": a detailed engineering-level description of what the mismatch is\n"
            "- \"location\": location on page (e.g., \"Header/Nav\", \"Hero section CTA\", \"Footer links\")\n"
            "- \"expected\": what the Figma mockup expects (dimensions, layout, color)\n"
            "- \"actual\": what the live website screenshot actually rendered\n\n"
            "Ensure the JSON output is valid and clean. Do not wrap the JSON in Markdown or backticks."
        )

        def call_gemini():
            response = self.model.generate_content(
                [system_prompt, mockup_img, screenshot_img],
                generation_config={"response_mime_type": "application/json"}
            )
            return response.text

        response_text = await loop.run_in_executor(None, call_gemini)

        try:
            data = json.loads(response_text)
            return data.get("findings", [])
        except Exception as e:
            cleaned_text = response_text.strip()
            if cleaned_text.startswith("```json"):
                cleaned_text = cleaned_text[7:]
            if cleaned_text.endswith("```"):
                cleaned_text = cleaned_text[:-3]
            try:
                data = json.loads(cleaned_text.strip())
                return data.get("findings", [])
            except Exception:
                print(f"Failed to parse Gemini output: {e}\nRaw Response: {response_text}")
                return [{
                    "category": "layout",
                    "severity": "High",
                    "finding": f"AI audit output failed to parse correctly. Raw text: {response_text[:200]}",
                    "location": "Global",
                    "expected": "Valid JSON findings list",
                    "actual": "Malformed AI response"
                }]

    async def _audit_groq(self, mockup_img: Image.Image, screenshot_img: Image.Image) -> list:
        # Convert images to base64 for API delivery
        def img_to_b64(img):
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
            
        mockup_b64 = img_to_b64(mockup_img)
        screenshot_b64 = img_to_b64(screenshot_img)
        
        system_prompt = (
            "You are a Senior QA Automation Engineer performing a category-by-category visual audit "
            "comparing a Figma mockup (Image 1) against a live website screenshot (Image 2).\n\n"
            "Analyze visual differences in the following categories:\n"
            "- layout (grid alignment, responsiveness structural columns)\n"
            "- component position (offsets, alignment, ordering)\n"
            "- sizing (height, width, button dimensions)\n"
            "- color (background, borders, branding color deviations)\n"
            "- typography (fonts, weights, sizes, styles)\n"
            "- spacing (margins, padding, row/column gaps)\n"
            "- missing/extra elements (headers, footers, buttons, icons)\n"
            "- styling details (border radius, box shadows, gradients)\n"
            "- images (missing graphic content, aspect ratio distortion)\n"
            "- responsive issues (text wrapping, overflow elements)\n\n"
            "You must return a JSON object with a single key \"findings\" which is a list of findings.\n"
            "Each finding in the list MUST be an object with the following fields:\n"
            "- \"category\": string matching one of the categories above\n"
            "- \"severity\": string value of exactly: \"Critical\", \"High\", \"Medium\", or \"Low\"\n"
            "- \"finding\": a detailed engineering-level description of what the mismatch is\n"
            "- \"location\": location on page (e.g., \"Header/Nav\", \"Hero section CTA\", \"Footer links\")\n"
            "- \"expected\": what the Figma mockup expects (dimensions, layout, color)\n"
            "- \"actual\": what the live website screenshot actually rendered\n\n"
            "Ensure the JSON output is valid and clean. Do not wrap the JSON in Markdown or backticks."
        )

        headers = {
            "Authorization": f"Bearer {self.groq_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "meta-llama/llama-4-scout-17b-16e-instruct",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": system_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{mockup_b64}"}},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"}}
                    ]
                }
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.2
        }

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
                response.raise_for_status()
                res_data = response.json()
                response_text = res_data["choices"][0]["message"]["content"]
                
            data = json.loads(response_text)
            return data.get("findings", [])
        except Exception as e:
            print(f"Error querying Groq vision API: {e}")
            return [{
                "category": "layout",
                "severity": "High",
                "finding": f"AI audit output failed to query Groq correctly. Error: {str(e)}",
                "location": "Global",
                "expected": "Valid Groq JSON response",
                "actual": "API Error"
            }]

    def calculate_score(self, findings: list) -> tuple[float, dict]:
        """
        Calculates severity-weighted deductions capped at 90.
        Critical = 25
        High = 12
        Medium = 4
        Low = 1
        """
        weights = {
            "Critical": 25.0,
            "High": 12.0,
            "Medium": 4.0,
            "Low": 1.0
        }

        severity_counts = {
            "Critical": 0,
            "High": 0,
            "Medium": 0,
            "Low": 0
        }

        total_deduction = 0.0

        for f in findings:
            sev = f.get("severity", "Low")
            sev_title = sev.title() if isinstance(sev, str) else "Low"
            if sev_title not in weights:
                sev_title = "Low"
            
            weight = weights[sev_title]
            total_deduction += weight
            severity_counts[sev_title] += 1

        capped_deduction = min(90.0, total_deduction)
        overall_score = 100.0 - capped_deduction

        breakdown = {
            "raw_total_deduction": total_deduction,
            "capped_deduction": capped_deduction,
            "severity_counts": severity_counts
        }

        return overall_score, breakdown
