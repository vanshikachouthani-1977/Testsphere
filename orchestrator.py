import asyncio
from typing import List, Dict, Any
# pyrefly: ignore [missing-import]
from PIL import Image

from agents.ingestion_agent import IngestionAgent
from agents.vector_embedding_agent import VectorEmbeddingAgent
from agents.pixel_comparison_agent import PixelComparisonAgent
from agents.ai_vision_agent import AIVisionAgent
from agents.test_case_mapper_agent import TestCaseMapperAgent
from agents.report_agent import ReportAgent

VECTOR_THRESHOLD = 0.65
PIXEL_THRESHOLD = 0.40

class QAOrchestrator:
    def __init__(self):
        self.ingestion_agent = IngestionAgent()
        self.vector_agent = VectorEmbeddingAgent()
        self.pixel_agent = PixelComparisonAgent()
        self.ai_agent = AIVisionAgent()
        self.mapper_agent = TestCaseMapperAgent()
        self.report_agent = ReportAgent()

    async def run_audit(
        self, 
        mockup_bytes: bytes, 
        screenshot_bytes: bytes, 
        test_cases: List[str]
    ) -> Dict[str, Any]:
        """
        Coordinates the full gatekeeper workflow:
        1. Ingests and normalizes mockup and screenshot.
        2. Runs Vector and Pixel checks in parallel.
        3. Evaluates thresholds. If either fails, applies flat 70% deduction and skips AI.
        4. Runs AI Vision audit & maps findings to test cases.
        5. Compiles final report.
        """
        # 1. Ingestion / Normalization
        mockup_img, screenshot_img = self.ingestion_agent.process_images(mockup_bytes, screenshot_bytes)
        
        # 2. Run check gatekeepers in parallel
        # Note: Vector embedding uses sentence-transformers/PyTorch which can block CPU; run in thread pool
        # SSIM calculation also runs in a thread pool to prevent blocking the event loop
        vector_task = asyncio.to_thread(self.vector_agent.compare, mockup_img, screenshot_img)
        pixel_task = asyncio.to_thread(self.pixel_agent.compare, mockup_img, screenshot_img)
        
        vector_similarity, (pixel_similarity, diff_img_bytes) = await asyncio.gather(vector_task, pixel_task)
        
        # 3. Threshold Decision Gate
        is_fundamental_mismatch = (vector_similarity < VECTOR_THRESHOLD) or (pixel_similarity < PIXEL_THRESHOLD)
        
        ai_findings = None
        deduction_breakdown = {}
        
        if is_fundamental_mismatch:
            reason = (
                f"Fundamental mismatch detected: "
                f"Vector Similarity ({vector_similarity:.2f}) < {VECTOR_THRESHOLD} OR "
                f"Pixel Similarity ({pixel_similarity:.2f}) < {PIXEL_THRESHOLD}"
            )
            deduction_breakdown = {
                "fundamental_mismatch_deduction": 70.0,
                "reason": reason
            }
            overall_score = 30.0
            
            # Map test cases against mismatch reasoning directly, skipping AI audit
            test_case_results = await self.mapper_agent.map_mismatch(test_cases, reason)
        else:
            # 4. AI Vision Agent Audit (runs only if threshold gate passes)
            ai_findings = await self.ai_agent.audit(mockup_img, screenshot_img)
            
            # Calculate deductions
            overall_score, deduction_breakdown = self.ai_agent.calculate_score(ai_findings)
            
            # Map findings to test cases
            test_case_results = await self.mapper_agent.map_findings(test_cases, ai_findings)
            
        # 5. Report Generation
        report = self.report_agent.generate(
            vector_similarity=vector_similarity,
            pixel_similarity=pixel_similarity,
            is_fundamental_mismatch=is_fundamental_mismatch,
            ai_findings=ai_findings,
            test_case_results=test_case_results,
            overall_similarity_score=overall_score,
            deduction_breakdown=deduction_breakdown,
            diff_image_bytes=diff_img_bytes
        )
        
        # Store diff_image_bytes on orchestrator so main.py can retrieve/save it
        self.last_diff_image_bytes = diff_img_bytes
        
        return report
