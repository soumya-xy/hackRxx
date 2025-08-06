from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from app.core.config import settings
import json
import re

class LLMService:
    """
    Service responsible for interacting with the Large Language Model (LLM)
    for query parsing and contextual decision-making.
    """
    def __init__(self):
        """
        Initializes the LLMService with the specified Gemini model.
        """
        self.llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", temperature=0)

    async def extract_structured_query(self, natural_query: str) -> str:
        """
        Uses the LLM to parse a natural language query into a structured format.

        Args:
            natural_query (str): The user's natural language question.

        Returns:
            str: A structured representation of the query's intent and conditions.
        """
        prompt_template = """
        You are a powerful query parser. Your task is to extract the core intent, entities, and conditions from a natural language query about a policy document.
        
        Natural Query: {natural_query}
        
        Please provide a structured, detailed query that can be used to search for relevant information.
        
        Example:
        Natural Query: "Does this policy cover knee surgery, and what are the conditions?"
        Structured Query: "POLICY COVERAGE: Knee surgery. CONDITIONS: Waiting period, sub-limits, exclusions."
        
        Natural Query: {natural_query}
        Structured Query: 
        """
        prompt = PromptTemplate(template=prompt_template, input_variables=["natural_query"])
        
        chain = prompt | self.llm
        response = chain.invoke({"natural_query": natural_query})
        
        return response.content

    async def evaluate_and_answer(self, structured_query: str, retrieved_clauses: list, question: str) -> dict:
        """
        Evaluates retrieved policy clauses against a structured query and a user question
        to formulate a final answer, source clauses, and rationale in JSON format.

        Args:
            structured_query (str): The parsed structured query.
            retrieved_clauses (list): A list of Document objects containing relevant text chunks.
            question (str): The original user question.

        Returns:
            dict: A dictionary containing the 'answer', 'source_clauses', and 'rationale'.
        """
        clause_text = "\n\n".join([doc.page_content for doc in retrieved_clauses])
        
        prompt_template = """
        You are an expert policy analyst. You have been given a user question, a structured query, and several relevant policy clauses.
        Your task is to answer the question based *only* on the provided clauses. If the clauses do not contain the answer, state that you cannot find the information.
        
        **CRITICAL INSTRUCTIONS:**
        1.  **Extract Specific Numerical Values:** When the question asks for quantities, durations, percentages, or monetary limits, you MUST find and state the exact numerical value from the clauses. Do not generalize or state that it's "subject to limits" if a specific number is present.
        2.  **Prioritize General Rules:** If a question asks for a general policy (e.g., "waiting period for pre-existing diseases"), look for overarching rules before specific examples. Only include specific examples if the general rule is not found, or if the question explicitly asks for examples.
        3.  **Maintain Specificity:** Ensure the answer directly addresses all parts of the question.
        
        User Question: {question}
        Structured Query: {structured_query}
        Relevant Clauses:
        {clauses}
        
        Your response must be in a JSON format with three keys:
        1. "answer": The direct answer to the question.
        2. "source_clauses": A list of the specific clauses that were used to formulate the answer.
        3. "rationale": A brief explanation of how you arrived at the answer using the clauses.
        
        Answer:
        """
        
        prompt = PromptTemplate(
            template=prompt_template, 
            input_variables=["question", "structured_query", "clauses"]
        )

        chain = prompt | self.llm
        response = chain.invoke({
            "question": question, 
            "structured_query": structured_query, 
            "clauses": clause_text
        })
        
        raw_response = response.content
        
        try:
            json_str = re.search(r'\{.*\}', raw_response, re.DOTALL).group(0)
            return json.loads(json_str)
        except (json.JSONDecodeError, AttributeError) as e:
            print(f"Error parsing JSON from LLM: {e}")
            return {
                "answer": "An error occurred while processing the response. The LLM's output was not in the expected JSON format.",
                "source_clauses": [],
                "rationale": f"Failed to parse LLM response into JSON. Raw response: {raw_response[:200]}..."
            }

