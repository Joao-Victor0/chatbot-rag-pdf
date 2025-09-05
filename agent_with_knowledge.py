import os
from langchain.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama, OllamaEmbeddings

class AgentWithKnowledge:
    def __init__(self):
        self.model = ChatOllama(model="gemma3:latest")
        self.embedding_function = OllamaEmbeddings(model="llama3")
        self.CHROMA_PATH = "./chroma_db"
        self.db = None


    def __setup_database(self): #importa a base de dados se já existir
        if os.path.exists(self.CHROMA_PATH):
            db = Chroma(
                persist_directory=self.CHROMA_PATH,
                embedding_function=self.embedding_function
            )

        return db
    

    def __setup_template(self): #define o template de pergunta e resposta do agent
        prompt_template = """
        Você é um assistente de IA especialista em analisar documentos e responder perguntas acerca do documento. 
        Responda as perguntas estritamente baseado no contexto fornecido pelo documento abaixo:

        Contexto: {context}
        Questão: {question}
        """

        prompt = ChatPromptTemplate.from_template(template=prompt_template)
        return prompt


    def __setup_knowledge_base(self): #Configura a base de conhecimento a partir do Chain 
        #Configurações
        self.db = self.__setup_database()
        prompt = self.__setup_template()
        retriever = self.db.as_retriever(search_kwargs={"k": 8}) #recupera as informacoes na base de dados

        #Conecta a sequência de ações (como um fluxo de trabalho)
        chain = ({"context": retriever, "question": RunnablePassthrough()} 
                 | prompt 
                 | self.model
        )

        return chain

    def ask(self, query:str):
        chain = self.__setup_knowledge_base()        
        response = chain.invoke(query)
        return response.content
        #results = self.db.similarity_search_with_relevance_scores(query=query, k=8)
        #return results
    
#agent = AgentWithKnowledge()
#response = agent.ask("Quais as iniciativas presentes na Portaria Normativa Nº 06 de 10 de Maio de 2022?")
#print(response)