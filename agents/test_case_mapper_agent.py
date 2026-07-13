import os
import json
import asyncio
import httpx
import google.generativeai as genai

class TestCaseMapperAgent:
    def __init__(self):
        # Load environment variables
        from dotenv import load_dotenv
        load_dotenv()
        
        self.gemini_key = os.getenv("GEMINI_API_KEY")
        self.groq_key = os.getenv("GROQ_API_KEY")
        
        if self.gemini_key:
            self.model = genai.GenerativeModel("gemini-1.5-flash")
            print("TestCaseMapperAgent initialized using Gemini.")
        else:
            print("TestCaseMapperAgent initialized using Groq LLM.")

    async def map_findings(self, test_cases: list[str], findings: list) -> list[dict]:
        """
        Uses Gemini or Groq to cross-reference user-defined test cases against visual findings.
        Returns a list of dictionaries with {test_case, status, reason}.
        """
        if not test_cases:
            return []

        if self.gemini_key:
            return await self._map_findings_gemini(test_cases, findings)
        else:
            return await self._map_findings_groq(test_cases, findings)

    async def _map_findings_gemini(self, test_cases: list[str], findings: list) -> list[dict]:
        loop = asyncio.get_running_loop()

        prompt = (
            "You are a QA Verification Agent. Your job is to match a list of user-defined test cases "
            "against findings discovered during a visual comparison audit between a design mockup and a screenshot.\n\n"
            f"User-Defined Test Cases:\n{json.dumps(test_cases, indent=2)}\n\n"
            f"Visual Audit Findings:\n{json.dumps(findings, indent=2)}\n\n"
            "For each testcase, evaluate whether the findings imply a status of:\n"
            "- \"PASS\": The findings contain no mismatches that violate this test case.\n"
            "- \"FAIL\": The findings explicitly describe a mismatch that violates this test case.\n"
            "- \"PARTIAL\": The findings suggest the element exists but deviates slightly from design guidelines.\n\n"
            "You must return a JSON object with a single key \"results\" which is a list of results.\n"
            "Each result in the list MUST contain the following fields:\n"
            "- \"test_case\": the exact string of the test case from the input list\n"
            "- \"status\": exactly \"PASS\", \"FAIL\", or \"PARTIAL\"\n"
            "- \"reason\": a concise explanation detailing why this status was selected using specific references from the findings.\n\n"
            "Ensure the JSON output is valid. Do not wrap the JSON in Markdown or backticks."
        )

        def call_gemini():
            response = self.model.generate_content(
                prompt,
                generation_config={"response_mime_type": "application/json"}
            )
            return response.text

        response_text = await loop.run_in_executor(None, call_gemini)

        try:
            data = json.loads(response_text)
            return data.get("results", [])
        except Exception as e:
            cleaned_text = response_text.strip()
            if cleaned_text.startswith("```json"):
                cleaned_text = cleaned_text[7:]
            if cleaned_text.endswith("```"):
                cleaned_text = cleaned_text[:-3]
            try:
                data = json.loads(cleaned_text.strip())
                return data.get("results", [])
            except Exception:
                print(f"Error parsing Gemini Test Case Mapper JSON: {e}\nRaw Response: {response_text}")
                return [{
                    "test_case": tc,
                    "status": "FAIL",
                    "reason": "Failed to run AI Test-Case Mapper on this item."
                } for tc in test_cases]

    async def _map_findings_groq(self, test_cases: list[str], findings: list) -> list[dict]:
        prompt = (
            "You are a QA Verification Agent. Your job is to match a list of user-defined test cases "
            "against findings discovered during a visual comparison audit between a design mockup and a screenshot.\n\n"
            f"User-Defined Test Cases:\n{json.dumps(test_cases, indent=2)}\n\n"
            f"Visual Audit Findings:\n{json.dumps(findings, indent=2)}\n\n"
            "For each testcase, evaluate whether the findings imply a status of:\n"
            "- \"PASS\": The findings contain no mismatches that violate this test case.\n"
            "- \"FAIL\": The findings explicitly describe a mismatch that violates this test case.\n"
            "- \"PARTIAL\": The findings suggest the element exists but deviates slightly from design guidelines.\n\n"
            "You must return a JSON object with a single key \"results\" which is a list of results.\n"
            "Each result in the list MUST contain the following fields:\n"
            "- \"test_case\": the exact string of the test case from the input list\n"
            "- \"status\": exactly \"PASS\", \"FAIL\", or \"PARTIAL\"\n"
            "- \"reason\": a concise explanation detailing why this status was selected using specific references from the findings.\n\n"
            "Ensure the JSON output is valid. Do not wrap the JSON in Markdown or backticks."
        )

        headers = {
            "Authorization": f"Bearer {self.groq_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.2
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
                response.raise_for_status()
                res_data = response.json()
                response_text = res_data["choices"][0]["message"]["content"]
                
            data = json.loads(response_text)
            return data.get("results", [])
        except Exception as e:
            print(f"Error querying Groq text mapping API: {e}")
            return [{
                "test_case": tc,
                "status": "FAIL",
                "reason": f"Failed to run Groq mapping: {str(e)}"
            } for tc in test_cases]

    async def map_mismatch(self, test_cases: list[str], reason: str) -> list[dict]:
        """
        Directly maps all test cases to FAIL when a fundamental mismatch has occurred.
        """
        return [
            {
                "test_case": tc,
                "status": "FAIL",
                "reason": f"Skipped AI audit and failed test case due to fundamental mismatch: {reason}"
            }
            for tc in test_cases
        ]
