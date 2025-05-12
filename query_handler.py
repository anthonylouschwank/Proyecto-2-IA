import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain.embeddings import OpenAIEmbeddings

load_dotenv()

class QueryHandler:
    def __init__(self):
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.embedder = OpenAIEmbeddings(model="text-embedding-ada-002")
    
    def get_embedding(self, text):
        return self.embedder.embed_query(text)
    
    def search(self, query, top_k=5):
        # Convertir pregunta a vector
        query_embedding = self.get_embedding(query)
        
        # Buscar en Pinecone
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        # Formatear resultados
        formatted_results = []
        for match in results.matches:
            formatted_results.append({
                "id": match.id,
                "score": match.score,
                "text": match.metadata["text"],
                "source": match.metadata.get("source", "N/A")
            })
        
        return formatted_results