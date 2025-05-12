import streamlit as st
from tools.embeds import EmbeddingManager
from tools.openai_tools import OpenAIResponseGenerator
import os
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Configuración de la pagina
st.set_page_config(
    page_title="Asistente IA - Asistente Inteligente",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cargar estilos CSS
def load_css(css_file):
    with open(css_file, 'r') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Cargar archivo CSS
css_path = os.path.join(os.path.dirname(__file__), "styles.css")
if os.path.exists(css_path):
    load_css(css_path)
else:
    st.warning("Archivo CSS no encontrado. Algunos estilos pueden no cargarse correctamente.")

# Inicializar componentes
emb_manager = EmbeddingManager()
openai_generator = OpenAIResponseGenerator()
index = emb_manager.get_index()

# Sidebar con opciones y branding
with st.sidebar:
        
    # Opciones
    st.subheader("Configuracion")
    show_context = st.checkbox("Mostrar contexto utilizado", value=True)
    show_scores = st.checkbox("Mostrar puntajes de relevancia", value=False)
    
    st.divider()
    
    # Sección para cargar nuevos documentos
    st.subheader("Añadir nuevo conocimiento")
    uploaded_file = st.file_uploader(
        "Sube un archivo (TXT o PDF)",
        type=["txt", "pdf"],
        accept_multiple_files=False
    )
    
    if uploaded_file:
        # Guardar archivo
        save_path = os.path.join("data", "user_uploads", uploaded_file.name)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success(f"Archivo {uploaded_file.name} guardado exitosamente.")
        st.info("Ejecuta load_documents.py para procesar el nuevo conocimiento.")
    
    st.divider()
    

# Contenido principal
col1, col2 = st.columns([2, 1])

with col1:
    # Encabezado 
    st.title("Asistente de Consulta")
    st.markdown("""
    <p style='font-size: 1.1rem; color: #97c1d1;'>
    Realiza preguntas técnicas y recibe respuestas detalladas y precisas.
    </p>
    """, unsafe_allow_html=True)
    
    # Input
    question = st.text_input(
        "Escribe tu pregunta:",
        placeholder="Ej: ¿Como afecta el fracking al medio ambiente?"
    )
    
    # Botón de búsqueda
    search_button = st.button("Buscar respuesta", use_container_width=True)
    
    if question and search_button:
        with st.spinner("Buscando respuesta..."):
            try:
                # 1. Obtener embedding de la pregunta
                query_embedding = emb_manager.get_embedding(question)
                
                # 2. Buscar en Pinecone
                results = index.query(
                    vector=query_embedding,
                    top_k=5,
                    include_metadata=True
                )
                
                # 3. Extraer textos relevantes
                context_texts = [match.metadata["text"] for match in results.matches]
                
                # 4. Generar respuesta usando OpenAI
                response = openai_generator.generate_response(question, context_texts)
                
                # 5. Mostrar respuesta principal primero
                st.subheader("Respuesta:")
                st.write(response)
                
                # 6. Mostrar contexto solo si está habilitado
                if show_context:
                    st.subheader("Contexto utilizado:")
                    for i, (text, match) in enumerate(zip(context_texts, results.matches)):
                        with st.expander(f"Contexto {i+1} (Score: {match.score:.2f})" if show_scores else f"Contexto {i+1}"):
                            st.write(text)
                
            except Exception as e:
                st.error(f"Error al procesar la pregunta: {str(e)}")
                st.exception(e)

with col2:
    # Twmas
    st.markdown("### TEMAS DEL CONOCIMIENTO:")
    
    # IA y Computación
    st.markdown("#### Temperatura y Computación")

    # Clima y Ecología
    st.markdown("#### Clima y Ecología")

    #Agricultura
    st.markdown("#### Agricultura")