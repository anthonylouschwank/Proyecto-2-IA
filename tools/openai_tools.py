from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()

class OpenAIResponseGenerator:
    def __init__(self):
        self.llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.7,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        self.prompt_template = """Eres un asistente tecnico experto. Tu rarea es proporcionar respuestas completas y detalladas a preguntas.

            Utiliza la siguiente informacion como contexto para responder a la pregunta que te haga el usuario.

            Si la informacion proporcionada no es suficiente para dar una respuesta completa, menciona que no puedes responder con certeza, pero proporciona lo que sepas al respecto.

            No copies textualmente el contexto, sintetiza y explica con diferentes palabras de manera didactica y profesional. 

            Estructurará tu respuesta con títulos y subtítulos cuando sea apropiado.
        
        Contexto:
        {context}
        
        Pregunta del usuario: {question}
        
        Tu respuesta completa y detallada:"""
        
        self.prompt = PromptTemplate(
            template=self.prompt_template,
            input_variables=["context", "question"]
        )
    
    def generate_response(self, question, context_texts):
        context = "\n\n".join(context_texts)
        
        chain = self.prompt | self.llm
        
        result = chain.invoke({
            "context": context,
            "question": question
        })
        
        # Resultado: un objeto
        return result.content