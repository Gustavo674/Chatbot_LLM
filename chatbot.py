from langchain_community.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
import gradio as gr

# Substitua com o caminho do PDF
pdf_path = "Engineering-workshop-health-and-safety-guidelines-catalog.pdf"

# Carregar o conteúdo do PDF
pdf_loader = PyPDFLoader(pdf_path)
documents = pdf_loader.load()
document_text = [doc.page_content for doc in documents]

# Configurar vetores com FAISS
embedding_model = OpenAIEmbeddings()  # Configure sua API key para OpenAI
vector_store = FAISS.from_texts(document_text, embedding_model)

# Configurar o modelo de linguagem (T5)
llm_pipeline = pipeline("text2text-generation", model="t5-small")
llm = HuggingFacePipeline(pipeline=llm_pipeline)

# Configurar a pipeline de QA com RAG
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vector_store.as_retriever())

# Função de RAG
def chatbot(input_text):
    try:
        response = qa_chain.run(input_text)
        return response
    except Exception as e:
        return f"An error occurred: {e}"

# Criar a interface Gradio
interface = gr.Interface(
    fn=chatbot,
    inputs="text",
    outputs="text",
    title="Workshop Safety Guidelines Chatbot (RAG)",
    description="Ask any questions about safety in workshop environments. Based on provided guidelines.",
)

# Lançar a interface
interface.launch()
