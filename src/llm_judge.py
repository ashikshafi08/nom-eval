import json
from typing import Dict, Any
from src.vllm_llm import vLLM_LLM

class LLMJudge:
    def __init__(self, llm: vLLM_LLM):
        self.llm = llm
        self.evaluation_prompt = """
        You are an expert judge evaluating responses to programming and code quality questions. Your task is to evaluate a given answer based on the question and any provided context using the criteria below.

        Evaluation Criteria (Additive Score, 0-3):
        1. Correctness: Award 1 point if the answer is technically correct and directly addresses the question.
        2. Completeness: Add 1 point if the answer covers all aspects of the question comprehensively.
        3. Code Quality / Best Practices: Add a final point if the answer demonstrates or suggests high-quality coding practices.

        Evaluation Steps:
        1. Carefully read the question, any provided context, and the answer.
        2. Assess each criterion individually, awarding points as appropriate.
        3. Provide detailed reasoning for each criterion, explaining your decision.
        4. Calculate the total score by summing the points awarded.
        5. Format your evaluation response according to the specified Output format.

        Output format:
        {{
          "reasoning": {{
            "correctness": "Explanation for correctness score",
            "completeness": "Explanation for completeness score",
            "code_quality": "Explanation for code quality score"
          }},
          "total_score": sum of criteria scores
        }}

        Question: {question}
        Context: {context}
        Answer: {answer}

        Provide your evaluation:
        """

    async def evaluate(self, question: str, answer: str, context: str = "") -> Dict[str, Any]:
        prompt = self.evaluation_prompt.format(question=question, context=context, answer=answer)
        response = await self.llm.async_query(prompt)
        return self._parse_evaluation(response)

    def _parse_evaluation(self, response: str) -> Dict[str, Any]:
        try:
            evaluation = json.loads(response)
            if not isinstance(evaluation, dict) or 'reasoning' not in evaluation or 'total_score' not in evaluation:
                raise ValueError("Invalid evaluation format")
            return evaluation
        except (json.JSONDecodeError, ValueError) as e:
            # Fallback parsing if JSON is invalid
            return {
                "reasoning": {
                    "correctness": "Unable to parse evaluation",
                    "completeness": "Unable to parse evaluation",
                    "code_quality": "Unable to parse evaluation"
                },
                "total_score": 0,
                "error": str(e)
            }

# Example usage
async def example_usage():
    llm = vLLM_LLM(model_id="mistralai/Mistral-7B-Instruct-v0.3")
    judge = LLMJudge(llm)

    question = "How can we improve the error handling in this Python function to make it more robust?"
    context = """
    def divide(a, b):
        return a / b
    """
    answer = """
    To improve error handling, we can use a try-except block:

    def divide(a, b):
        try:
            result = a / b
        except ZeroDivisionError:
            print("Error: Division by zero!")
            return None
        except TypeError:
            print("Error: Invalid input types!")
            return None
        return result

    This handles division by zero and type errors, making the function more robust.
    """

    evaluation = await judge.evaluate(question, answer, context)
    print(json.dumps(evaluation, indent=2))

if __name__ == "__main__":
    import asyncio
    asyncio.run(example_usage())
