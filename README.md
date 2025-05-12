# Proyecto 2 de Inteligencia Artificial

Este proyecto corresponde al segundo proyecto del curso de Inteligencia Artificial.

## 🚀 Instrucciones para correr el proyecto

1. **Instala las dependencias:**

   > Asegúrate de tener un entorno virtual activado (opcional pero recomendado).

   ```bash
   pip install -r requirements.txt

2. **Corre las siguientes lineas:**

   > En el orden que estan.

   ```bash
   python load_documents.py
   streamlit run app.py
   
##  Descripción de Módulos

###  `data/`
Contiene el contexto que se le proporciona a la IA.  
Incluye:
- Una carpeta con los documentos que el usuario inserta.
- Otra carpeta con información adicional agregada manualmente, pensada para ofrecer una base de conocimiento más amplia para ejemplos y pruebas.

---

###  `tools/`
Contiene las herramientas necesarias para el funcionamiento del sistema, incluyendo:
- Envío y manejo de información hacia las distintas APIs.
- El *prompt* principal que se utiliza para consultar a ChatGPT, junto con el contexto correspondiente.
- Generación de *embeddings*, creación del índice del proyecto, e inicialización de **Pinecone** y **ChatGPT**.

---

### 🖥️ Página Principal (`app.py`)
Es la interfaz principal del proyecto.  
Utiliza **Streamlit** para generar la estructura HTML de la aplicación web, incluyendo las secciones donde se recibe y muestra la información generada por el modelo.
