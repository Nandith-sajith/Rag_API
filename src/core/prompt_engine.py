from groq import AsyncGroq  # Use AsyncGroq for async 
from src.core.config import settings
from src.core.utils import calculate_confidence_score
from typing import List, Dict, Tuple

class PromptEngine:
    def __init__(self):
        self.groq_client = AsyncGroq(api_key=settings.groq_api_key)  # Async client
        self.model = "llama3-8b-8192"
        self.max_tokens = 500

        self.system_message = {
            "role": "system",
            "content": (
                "You are an expert assistant answering questions based solely on provided context from Monopoly rules. "
                "Your responses must be concise, accurate, and strictly derived from the context—do not hallucinate or add "
                "external information. Cite page numbers from the context where applicable. For complex queries, reason "
                "step-by-step. If the context is insufficient, respond only with: 'I don’t have enough information to "
                "answer that based on the Monopoly rules provided.'"
            )
        }

        self.few_shot_examples = [
            {
                "role": "user",
                "content": (
                    "Query: How do you move in Monopoly?\n"
                    "Context: Players take turns rolling two dice to determine how many spaces to move on the board (Page 3).\n"
                    "Answer: In Monopoly, you move by rolling two dice, and the total determines how many spaces you advance (Page 3)."
                )
            },
            {
                "role": "user",
                "content": (
                    "Query: What happens when you land on a property?\n"
                    "Context: If a property is unowned, you may buy it from the Bank (Page 4). If owned, pay rent to the owner.\n"
                    "Answer: When you land on a property, you can buy it if unowned (Page 4), or pay rent if it’s owned."
                )
            }
        ]

    def build_prompt(self, query: str, context: str) -> List[Dict[str, str]]:
        """Construct prompt with anti-hallucination instructions."""
        messages = [self.system_message] + self.few_shot_examples
        user_message = {
            "role": "user",
            "content": (
                f"Query: {query}\n"
                f"Context: {context}\n"
                "Answer: Provide a concise, accurate response based only on the context. Cite page numbers where applicable. "
                "For complex queries, reason step-by-step. If the context lacks sufficient information, state: "
                "'I don’t have enough information to answer that based on the Monopoly rules provided.'"
            )
        }
        messages.append(user_message)
        return messages

    async def generate_answer(self, query: str, context: str, keywords: List[str]) -> Tuple[str, float]:
        """Generate answer asynchronously with confidence score."""
        messages = self.build_prompt(query, context)
        try:
            response = await self.groq_client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens
            )
            answer = response.choices[0].message.content.strip().replace('\n\n', '\n')
            confidence = calculate_confidence_score(context, answer, keywords)
        except Exception as e:
            answer = "I don’t have enough information to answer that based on the Monopoly rules provided." if not context else f"Error: {str(e)}"
            confidence = 0.0
        return answer, confidence

    def evaluate_response(self, answer: str, context: str, keywords: List[str]) -> Dict[str, float]:
        """Evaluate response quality and accuracy."""
        confidence = calculate_confidence_score(context, answer, keywords)
        coherence = 1.0 if answer and len(answer.split()) > 5 else 0.5
        accuracy = confidence
        return {
            "confidence": confidence,
            "coherence": coherence,
            "accuracy": accuracy
        }