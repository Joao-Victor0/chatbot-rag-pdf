from extract_pdf_to_json import PdfToJson
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.schema import Document
from langchain_core.documents import Document

import os
import re
import json

DIRECTORY = "./temp_files"
JSON_DIRECTORY = "./output"

def create_db():
    documents = load_json_document()
    chunks = split_json_chunks(documents=documents)
    vectorize_json_chunks(chunks)


def document_convert(file_path, data):
    docs_in_file = []

    #Tenta extrair o título/ID da primeira página
    document_title = "Título Padrão ou Nome do Arquivo" #um fallback (plano B)
    first_page_text = data.get("page_1", {}).get("text", "")

    #Tenta encontrar um padrão como "PORTARIA NORMATIVA N° XY"
    match = re.search(r"(PORTARIA NORMATIVA.*?Nº\s*\d+)", first_page_text, re.IGNORECASE)
    if match:
        document_title = match.group(1).strip()
    
    for page_key, page_content in data.items(): #itera sobre cada página do arquivo
        text = page_content.get("text", "") #extrai o texto limpo da página
        if not text:
            continue

        try: #extrai o número da página
            page_number = int(page_key.split("_")[1])

        except (IndexError, ValueError):
            page_number = None #lida com chaves mal formatadas se necessário

        metadata = {
            "source": file_path,
            "page": page_number,
            "title": document_title
        }

        doc_in_file = Document(page_content=text, metadata=metadata) #cria um arquivo do tipo Document para cada página do JSON
        docs_in_file.append(doc_in_file)

    return docs_in_file
       

def load_json_document():
    #Extrai e converte PDFs em JSON
    pdf_paths = os.listdir(DIRECTORY) #lista todos os PDFs
    for index, pdf_path in enumerate(pdf_paths):
        PdfToJson().extract_pdf_content(pdf_path=os.path.join(DIRECTORY, pdf_path), index=index)

    #Carrega os JSONs
    documents = []
    json_paths = [file for file in os.listdir(JSON_DIRECTORY) if file.endswith('.json')] #lista os arquivos .json e ignora o resto
    for json_path in json_paths:
        file_path = os.path.join(JSON_DIRECTORY, json_path) #cria um caminho único juntando o nome do diretório com o nome do arquivo
        
        with open(file_path, 'r', encoding="utf-8") as fp:
            data = json.load(fp)

        document = document_convert(file_path=file_path, data=data)
        documents.append(document) #adiciona o documento a uma lista de documentos

    return documents


def split_json_chunks(documents):
    #Quebra os documentos em Chunks
    documents_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, #tamanho de cada chunk
        chunk_overlap=150, #sobrepõe chunks, voltando 2500 chunks a partir do novo chunk, evitando perda de contexto
        length_function=len, #tamanho de cada chunk
        add_start_index=True
    )

    #Pega a página de cada documento da lista de documentos e une todas elas em uma única lista de páginas identificadas por arquivo
    joined_documents = [page_document for document in documents for page_document in document]

    chunks = documents_splitter.split_documents(documents=joined_documents) #cria chunks para cada documento e guarda todos juntos
    return chunks


def vectorize_json_chunks(chunks): #cria vetores numéricos com os chunks (para futuramente comparar a resposta do usuário com os números) 
    embedding_function = OllamaEmbeddings(model='mxbai-embed-large') #pega o modelo que realiza os embeddings
    Chroma.from_documents(documents=chunks, embedding=embedding_function, persist_directory="chroma_db") #cria o banco de dados vetorial

create_db()