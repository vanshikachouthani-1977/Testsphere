class ReportAgent:
    def __init__(self):
        pass

    def generate(
        self,
        vector_similarity: float,
        pixel_similarity: float,
        is_fundamental_mismatch: bool,
        ai_findings: list | None,
        test_case_results: list,
        overall_similarity_score: float,
        deduction_breakdown: dict,
        diff_image_bytes: bytes
    ) -> dict:
        """
        Compiles all comparison metrics, visual findings, and mapped test cases
        into a unified QA report with a final deployment recommendation.
        """
        # Determine deployment recommendation
        if is_fundamental_mismatch:
            recommendation = "No-go"
        elif overall_similarity_score >= 90.0:
            recommendation = "Go"
        elif overall_similarity_score >= 70.0:
            recommendation = "Go with fixes"
        else:
            recommendation = "No-go"

        # Compile the report dictionary
        report = {
            "metrics": {
                "vector_similarity": round(vector_similarity, 4),
                "pixel_similarity": round(pixel_similarity, 4),
                "is_fundamental_mismatch": is_fundamental_mismatch
            },
            "overall_similarity_score": round(overall_similarity_score, 2),
            "deduction_breakdown": deduction_breakdown,
            "deployment_recommendation": recommendation,
            "ai_findings": ai_findings,
            "test_case_results": test_case_results
        }

        return report
