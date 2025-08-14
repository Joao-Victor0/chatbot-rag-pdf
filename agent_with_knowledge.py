import os

from extract_pdf_to_json import PdfToJson
from langchain.prompts import ChatPromptTemplate

from langchain_chroma import Chroma
from langchain_community.document_loaders import JSONLoader

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

from langchain_ollama import ChatOllama, OllamaEmbeddings

class AgentWithKnowledge:
    def __init__(self):
        self.model = ChatOllama(model="deepseek-r1:latest")
        self.retriever = None
        self.retriever_chain = None

        #print("\n[] [DEBUG] Classe AgentWithKnowledge está sendo instanciada (só deve acontecer junto com o cache).")

    def setup_knowledge_base(self, pdf_path: str):
        embedding_function = OllamaEmbeddings(model="llama3")

        json_output_path="output/extracted_content.json"
        chroma_db_path="./chroma_db"

        #Vector Base, Embeddings and Retriever
        if os.path.exists(chroma_db_path):
            #print("⚡️ [DEBUG] CAMINHO RÁPIDO: Encontrou `chroma_db` e está carregando do disco.")
            db = Chroma(persist_directory=chroma_db_path, embedding_function=embedding_function)

        else:
            #PDF To JSON
            #print("🐢 [DEBUG] CAMINHO LENTO: Não encontrou `chroma_db`. Criando um novo banco de dados (ETAPA LENTA).")
            os.makedirs(os.path.dirname(json_output_path), exist_ok=True)
            if not os.path.exists(json_output_path):
                PdfToJson.extract_pdf_content(pdf_path=pdf_path)

            #Loading Documents
            loader = JSONLoader(file_path=json_output_path, jq_schema='.', text_content=False)
            documents = loader.load()

            #Create and Persist Vector Base
            db = Chroma.from_documents(
                documents=documents,
                embedding=embedding_function,
                persist_directory=chroma_db_path
            )

        self.retriever = db.as_retriever()
        #print("✅ [DEBUG] Retriever foi configurado.")

        #Context and Question
        system_template = """Você é um assistente de IA especialista em analisar documentos e responder perguntas acerca
        do documento. 
        Se a informação não estiver no contexto, diga 'Não encontrei a resposta no documento'
        Responda as perguntas estritamente baseado no contexto fornecido pelo documento abaixo:

        Contexto: {context}
        """

        human_template = "Questão: {input}"

        #Prompt and Model
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_template), 
            ("human", human_template)
        ])

        #Document and Retrieval Chains
        document_chain = create_stuff_documents_chain(self.model, prompt)
        self.retrieval_chain = create_retrieval_chain(self.retriever, document_chain)


    def ask(self, query:str):
        #Response
        #print(f"💬 [DEBUG] Método `ask` chamado com a query: '{query}'")
        if not self.retrieval_chain:
            return "Erro: A base de conhecimento não foi configurada. Chame o método setup_knowledge_base primeiro."
        
        response = self.retrieval_chain.invoke({"input": query})
        return response.get('answer', "Não foi possível encontrar uma resposta.")