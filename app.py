from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import os
import gradio as gr
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.gemini import Gemini

# loading environment variables if any
load_dotenv()

def setup_llm(api_key):
    # setting up gemini as our AI brain
    os.environ["GOOGLE_API_KEY"] = api_key
    Settings.llm = Gemini(model="models/gemini-2.5-flash", api_key=api_key)
    # using huggingface for free local embeddings
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

def process_document(file, api_key):
    # basic checks first - no point going further without these
    if not api_key:
        return None, "❌ Please enter your Gemini API key!"
    
    if file is None:
        return None, "❌ Please upload a document first!"
    
    try:
        setup_llm(api_key)
        
        # saving the uploaded file temporarily so llama-index can read it
        temp_dir = "temp_docs"
        os.makedirs(temp_dir, exist_ok=True)
        
        file_path = os.path.join(temp_dir, os.path.basename(file.name))
        with open(file_path, "wb") as f:
            with open(file.name, "rb") as src:
                f.write(src.read())
        
        # this is where the magic happens - reading and indexing the document
        # VectorStoreIndex converts text into vectors so AI can search through it
        documents = SimpleDirectoryReader(temp_dir).load_data()
        index = VectorStoreIndex.from_documents(documents)
        query_engine = index.as_query_engine()
        
        return query_engine, "✅ Document processed! Go ahead and ask anything."
    
    except Exception as e:
        return None, f"❌ Something went wrong: {str(e)}"

def answer_question(question, query_engine, api_key):
    # making sure everything is ready before answering
    if query_engine is None:
        return "❌ Upload a document first!"
    
    if not question:
        return "❌ Please type a question!"
    
    try:
        # querying the document - AI finds relevant parts and answers
        response = query_engine.query(question)
        return str(response)
    except Exception as e:
        return f"❌ Error while answering: {str(e)}"

# building the UI with gradio - keeps it simple and clean
with gr.Blocks(title="📄 Document Q&A App", theme=gr.themes.Soft()) as app:
    gr.Markdown("# 📄 Document Q&A App")
    gr.Markdown("*Upload any document — PDF, TXT, DOCX — and ask questions about it using Google Gemini AI!*")
    gr.Markdown("---")
    
    # storing the query engine in state so it persists between button clicks
    query_engine_state = gr.State(None)
    
    with gr.Row():
        api_key_input = gr.Textbox(
            label="🔑 Gemini API Key",
            placeholder="Paste your Gemini API key here...",
            type="password"
        )
    
    with gr.Row():
        file_input = gr.File(
            label="📁 Upload Your Document",
            file_types=[".pdf", ".txt", ".docx"]
        )
        upload_btn = gr.Button("📤 Process Document", variant="primary")
    
    status_output = gr.Textbox(label="Status", interactive=False)
    
    gr.Markdown("### Ask anything about your document 👇")
    
    with gr.Row():
        question_input = gr.Textbox(
            label="❓ Your Question",
            placeholder="e.g. What are the attendance rules? / Summarize this document",
            lines=2
        )
        ask_btn = gr.Button("🔍 Get Answer", variant="primary")
    
    answer_output = gr.Textbox(label="💡 Answer", lines=6, interactive=False)
    
    gr.Markdown("---")
    gr.Markdown("*Built by Suhani Saxena — because reading long documents manually is painful 😅*")
    
    upload_btn.click(
        fn=process_document,
        inputs=[file_input, api_key_input],
        outputs=[query_engine_state, status_output]
    )
    
    ask_btn.click(
        fn=answer_question,
        inputs=[question_input, query_engine_state, api_key_input],
        outputs=[answer_output]
    )

if __name__ == "__main__":
    app.launch()