import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import gradio as gr

# Configure sua API Key da OpenAI
os.environ["OPENAI_API_KEY"] = "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"

# Substitua com o caminho do PDF
pdf_path = "Engineering-workshop-health-and-safety-guidelines-catalog.pdf"

# Carregar o conteúdo do PDF
pdf_loader = PyPDFLoader(pdf_path)
documents = pdf_loader.load()
document_text = [doc.page_content for doc in documents]

# Configurar vetores com FAISS
embedding_model = OpenAIEmbeddings()  # API key já configurada acima
vector_store = FAISS.from_texts(document_text, embedding_model)

# Configurar o modelo de linguagem OpenAI
llm = ChatOpenAI(model="gpt-3.5-turbo")  # Use "text-davinci-003" se preferir o modelo completivo

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
