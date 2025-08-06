import asyncio
from app.services.document_parser import DocumentParser
from app.services.embedding_service import EmbeddingService
from app.services.llm_service import LLMService

class QueryProcessor:
    """
    Orchestrates the entire query-retrieval pipeline,
    including document parsing, embedding, search, LLM processing,
    and parallel question handling.
    """
    def __init__(self):
        self.doc_parser = DocumentParser()
        self.embedding_service = EmbeddingService()
        self.llm_service = LLMService()

    async def _process_single_question_async(self, question: str) -> dict:
        """
        Helper method to process a single question asynchronously.
        This encapsulates the LLM parsing, embedding search, and logic evaluation
        for one question.
        """
        # Step 3: LLM Parser - Extract structured query
        structured_query = await self.llm_service.extract_structured_query(question)
        
        # Step 4: Embedding Search - Retrieve similar clauses
        retrieved_clauses = await self.embedding_service.search_similar_clauses(structured_query)
        
        # Step 5: Logic Evaluation & Decision Making
        answer_json = await self.llm_service.evaluate_and_answer(structured_query, retrieved_clauses, question)
        
        # Step 6: JSON Output (detailed for internal use, simplified later)
        return answer_json

    async def run_pipeline(self, document_url: str, questions: list) -> dict:
        """
        Runs the entire LLM-powered query-retrieval pipeline.
        
        Args:
            document_url (str): The URL of the document to process.
            questions (list): A list of natural language questions to answer.

        Returns:
            dict: A dictionary containing the answers to the questions.
        """
        # Step 1: Parse Input Documents (still sequential, as it's a single document)
        document_text = self.doc_parser.parse_document(document_url)
        
        # Step 2: Embed the document for search (still sequential, as it's for the whole document)
        await self.embedding_service.upsert_document(document_text) # upsert_document should also be async

        # Parallelize the processing of each question
        # Create a list of coroutines (async tasks) for each question
        tasks = [self._process_single_question_async(question) for question in questions]

        # Run all tasks concurrently and wait for them to complete
        all_answers_detailed = await asyncio.gather(*tasks)
        
        # Format the response to match the hackathon's expected output (list of strings)
        simplified_answers = [item.get("answer", "No answer found.") for item in all_answers_detailed]
        
        return {"answers": simplified_answers}

