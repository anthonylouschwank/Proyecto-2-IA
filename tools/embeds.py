import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec  # Changed from PodSpec to ServerlessSpec
from langchain_openai import OpenAIEmbeddings

# Cargar variables de entorno
load_dotenv()

class EmbeddingManager:
    def __init__(self):
        self.api_key = os.getenv("PINECONE_API_KEY")
        self.index_name = os.getenv("PINECONE_INDEX_NAME")
        self.embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")
        
        if not self.api_key or not self.index_name:
            raise ValueError("Faltan variables de entorno PINECONE_API_KEY o PINECONE_INDEX_NAME")
        
        self.pc = Pinecone(api_key=self.api_key)
        
    def get_index(self):
        return self.pc.Index(self.index_name)
    
    def create_index_if_not_exists(self):
        if self.index_name not in self.pc.list_indexes().names():
            print(f"Creando índice '{self.index_name}'...")
            
            spec = ServerlessSpec(
                cloud="aws",
                region="us-east-1"  # AWS us-east-1 region
            )
            
            self.pc.create_index(
                name=self.index_name,
                dimension=1536,  
                metric="cosine",
                spec=spec
            )
            
            import time
            while not self.pc.describe_index(self.index_name).status['ready']:
                time.sleep(1)
            
            print("Indice creado.")
        else:
            print(f"El indice '{self.index_name}' ya existe.")
    
    def get_embedding(self, text):
        return self.embedding_model.embed_query(text)