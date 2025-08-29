from extract_pdf_to_json import PdfToJson
from langchain_community.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

import os

DIRECTORY = "./temp_files"
JSON_DIRECTORY = "./output"

def create_db():
    documents = load_documents()
    chunks = split_chunks(documents=documents)
    vectorize_chunks(chunks)


def load_documents():
    #Extrai e converte PDFs em JSON
    pdf_paths = os.listdir(DIRECTORY) #lista todos os PDFs
    for index, pdf_path in enumerate(pdf_paths):
        PdfToJson.extract_pdf_content(pdf_path=os.path.join(DIRECTORY, pdf_path), index=index)

    #Carrega os JSONs
    json_paths = [file for file in os.listdir(JSON_DIRECTORY) if file.endswith('.json')] #lista os arquivos .json e ignora o resto
    for json_path in json_paths:
        file_path = os.path.join(JSON_DIRECTORY, json_path) #cria um caminho único juntando o nome do diretório com o nome do arquivo
        loader = JSONLoader(file_path=file_path, jq_schema=".", text_content=False) #cria um carregador os arquivos JSON, somente
    documents = loader.load() #carrega os arquivos JSON e guarda em uma variável documents

    return documents


def split_chunks(documents):
    #Quebra os documentos em Chunks
    documents_splitter = RecursiveCharacterTextSplitter(
        chunk_size=5000, #tamanho de cada chunk
        chunk_overlap=2500, #sobrepõe chunks, voltando 2500 chunks a partir do novo chunk, evitando perda de contexto
        length_function=len, #tamanho de cada chunk
        add_start_index=True
    ) #provável problema em alguns dos parâmetros, pois o modelo só consegue responder até certo ponto de informação

    chunks = documents_splitter.split_documents(documents=documents)
    return chunks


def vectorize_chunks(chunks): #cria vetores numéricos com os chunks (para futuramente comparar a resposta do usuário com os números) 
    embedding_function = OllamaEmbeddings(model='llama3:latest') #pega o modelo que realiza os embeddings
    Chroma.from_documents(documents=chunks, embedding=embedding_function, persist_directory="chroma_db")

create_db()