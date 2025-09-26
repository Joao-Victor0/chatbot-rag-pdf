from extract_pdf_to_json import PdfToJson
from langchain_community.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import RecursiveJsonSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.schema import Document

import os
import unicodedata
import re
import json

DIRECTORY = "./temp_files"
JSON_DIRECTORY = "./output"

def create_db():
    documents = load_documents()
    #chunks = split_chunks(documents=documents)
    #vectorize_chunks(chunks)


def clean_data(document):
    #Decodifica os espapes unicode
    clean_text = document[0].page_content.encode("utf-8").decode("unicode_escape")

    #Normaliza os caracteres removendo as variações
    clean_text = unicodedata.normalize("NFKC", clean_text)

    #Retira quebras de linha excessivas 
    clean_text = re.sub(r"\s+", " ", clean_text)

    #Cria um documento limpo
    clean_document = Document(
        page_content=clean_text,
        metadata=document[0].metadata
    )

    return clean_document
    
    

def load_documents():
    #Extrai e converte PDFs em JSON
    pdf_paths = os.listdir(DIRECTORY) #lista todos os PDFs
    for index, pdf_path in enumerate(pdf_paths):
        PdfToJson().extract_pdf_content(pdf_path=os.path.join(DIRECTORY, pdf_path), index=index)

    #Carrega os JSONs
    documents = []
    json_paths = [file for file in os.listdir(JSON_DIRECTORY) if file.endswith('.json')] #lista os arquivos .json e ignora o resto
    for json_path in json_paths:
        file_path = os.path.join(JSON_DIRECTORY, json_path) #cria um caminho único juntando o nome do diretório com o nome do arquivo
        loader = JSONLoader(file_path=file_path, jq_schema=".", text_content=False) #cria um carregador os arquivos JSON, somente
        document = loader.load() #carrega o arquivo JSON
        clean_document = clean_data(document=document) #limpa o arquivo JSON

        documents.append(clean_document) #adiciona o documento a uma lista de documentos

    print(documents)
    return documents


def split_chunks(documents):
    #Quebra os documentos em Chunks
    documents_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400, #tamanho de cada chunk
        chunk_overlap=50, #sobrepõe chunks, voltando 2500 chunks a partir do novo chunk, evitando perda de contexto
        length_function=len, #tamanho de cada chunk
        add_start_index=True
    ) 

    chunks = documents_splitter.split_documents(documents=documents) #cria chunks para cada documento e guarda todos juntos
    return chunks


def vectorize_chunks(chunks): #cria vetores numéricos com os chunks (para futuramente comparar a resposta do usuário com os números) 
    embedding_function = OllamaEmbeddings(model='llama3:latest') #pega o modelo que realiza os embeddings
    Chroma.from_documents(documents=chunks, embedding=embedding_function, persist_directory="chroma_db")


create_db()

#--------------------------------------------------------------------------------------------------#
#                                   - OUTRA ABORDAGEM -
# -------------------------------------------------------------------------------------------------#


def clean_json(json_data):
    if isinstance(json_data, dict): #se o arquivo JSON for do tipo dicionário
        return {key: clean_json(value) for key, value in json_data.items()}

    elif isinstance(json_data, list): #se o arquivo JSON for do tipo lista
        return [clean_json(item) for item in json_data]
    
    elif isinstance(json_data, str): #se o arquivo JSON for uma string
        clean_text = unicodedata.normalize("NFKC", json_data) #normaliza os caracteres removendo as variações
        clean_text = re.sub(r"\s+", " ", clean_text) #retira quebras de linha excessivas 

        return clean_text
    
    else:
        return json_data


def load_json_documents():
    #Extrai e converte PDFs em JSON
    pdf_paths = os.listdir(DIRECTORY) #lista todos os PDFs
    for index, pdf_path in enumerate(pdf_paths):
        PdfToJson().extract_pdf_content(pdf_path=os.path.join(DIRECTORY, pdf_path), index=index)

    #Carrega os JSONs
    json_datas = []
    json_paths = [file for file in os.listdir(JSON_DIRECTORY) if file.endswith('.json')] #lista os arquivos .json e ignora o resto

    for json_path in json_paths:
        file_path = os.path.join(JSON_DIRECTORY, json_path) #cria um caminho único juntando o nome do diretório com o nome do arquivo

        with open(file_path, 'r', encoding='utf-8') as file:
            json_data = json.load(fp=file) #cria um carregador os arquivos JSON, somente

        clean_document = clean_json(json_data=json_data) #limpa o arquivo JSON
        json_datas.append(clean_document) #adiciona o documento a uma lista de documentos

    return json_datas
    

def split_json_chunks(json_datas):
    #Quebra os documentos em Chunks respeitando a estrutura de JSON
    json_splitter = RecursiveJsonSplitter(
        max_chunk_size=300 #tamanho máximo de cada chunk
    )

    json_chunks = []
    for json_data in json_datas:
        json_chunks.extend(json_splitter.split_json(json_data=json_data)) #cria chunks para cada documento e guarda todos juntos

    #Transformando os chunks no tipo Document
    document_chunks = []
    for chunk in json_chunks:
        #As informações do chunk para uma string que contém o conteúdo
        page_content = json.dumps(chunk, ensure_ascii=False, indent=2) #parametros de mantimento de caracteres especiais e legibilidade

        #Dicionário de Metadados
        metadata = {}

        #Criando um Document
        document = Document(
            page_content=page_content,
            metadata=metadata
        )

        document_chunks.append(clean_data(document=document))

    return document_chunks