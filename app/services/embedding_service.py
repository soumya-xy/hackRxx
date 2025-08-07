import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from app.core.config import settings
from pinecone import Pinecone, ServerlessSpec
import google.generativeai as genai

# Set the environment variables for Google
os.environ["GOOGLE_API_KEY"] = settings.GOOGLE_API_KEY

# Pinecone setup remains the same
os.environ["PINECONE_API_KEY"] = settings.PINECONE_API_KEY
pc = Pinecone(api_key=settings.PINECONE_API_KEY)
index_name = "hackathon-index"

# Configure the Gemini API client
genai.configure(api_key=settings.GOOGLE_API_KEY)

class EmbeddingService:
    """
    Service responsible for handling document embedding, Pinecone interactions,
    and reranking of search results.
    """
    def __init__(self):
        """
        Initializes the EmbeddingService with embedding model and text splitter.
        """
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200, 
            separators=["\n\n", "\n", ". ", "; ", " ", ""]
        )
        
        self.llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", temperature=0)

    async def upsert_document(self, document_text: str):
        """
        Chunks a document, generates embeddings, and upserts them into Pinecone.
        If the index does not exist, it will be created.
        """
        # Correctly check if the index exists
        active_indexes = pc.list_indexes().indexes
        if index_name not in [index.name for index in active_indexes]:
            pc.create_index(
                name=index_name,
                dimension=768,
                metric='cosine',
                spec=ServerlessSpec(cloud='aws', region='us-east-1')
            )
        
        docs = self.text_splitter.create_documents([document_text])
        # PineconeVectorStore.from_documents can be an async operation
        docsearch = await PineconeVectorStore.afrom_documents(docs, self.embeddings, index_name=index_name)
        return docsearch

    async def search_similar_clauses(self, query: str, top_k: int = 5):
        """
        Performs a similarity search in Pinecone.

        Args:
            query (str): The query string for semantic search.
            top_k (int): The number of top documents to retrieve from Pinecone.

        Returns:
            list: A list of top-ranked Document objects (chunks).
        """
        docsearch = PineconeVectorStore.from_existing_index(index_name, self.embeddings)
        
        # Retrieve documents asynchronously
        retrieved_docs = await docsearch.asimilarity_search(query, k=top_k)

        # Return the top documents directly (no reranking)
        return retrieved_docs

