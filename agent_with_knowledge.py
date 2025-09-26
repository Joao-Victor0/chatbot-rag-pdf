import os
from langchain.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama, OllamaEmbeddings

class AgentWithKnowledge:
    def __init__(self):
        self.model = ChatOllama(model="gemma3:latest")
        self.embedding_function = OllamaEmbeddings(model="mxbai-embed-large")
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
    

    def __setup_filter_query_extraction(self, query:str) -> str: #Usa uma LLM para obter o identificador do documento com base no query
        prompt_template = """
        Você é um assistente de IA especialista em análise de texto, agindo como um roteador de queries. Sua única tarefa é ler a pergunta do usuário e extrair o identificador completo e exato de um documento.

        O formato que eu preciso que você retorne é sempre 'PORTARIA NORMATIVA GR/UFRB Nº XX', onde XX é o número do documento.

        Se a pergunta não mencionar um documento específico, responda com a string 'N/A'. Não adicione nenhuma outra palavra ou explicação.

        Exemplos:
        Pergunta: "O que a portaria nº 06 diz sobre o uso de máscaras?"
        Sua Resposta: PORTARIA NORMATIVA GR/UFRB Nº 06

        Pergunta: "Me fale sobre as regras da portaria normativa 7 da UFRB."
        Sua Resposta: PORTARIA NORMATIVA GR/UFRB Nº 07

        Pergunta: "Quais documentos falam sobre inovação?"
        Sua Resposta: N/A

        Agora, analise a seguinte pergunta: "{query}"
        """

        prompt = prompt_template.format(query=query)
        filter_response = self.model.invoke(prompt).content
        return filter_response

    def ask(self, query:str):
        chain = self.__setup_knowledge_base()        
        #response = chain.invoke(query)
        #return response.content

        document_filter = self.__setup_filter_query_extraction(query=query)

        if document_filter != "N/A":
            results = self.db.similarity_search_with_relevance_scores(query=query, k=4, filter={"title": document_filter})

        else:
            results = self.db.similarity_search_with_relevance_scores(query=query, k=8)

        return results
    
agent = AgentWithKnowledge()
response = agent.ask("Quais as iniciativas presentes na PORTARIA NORMATIVA Nº 05 de 22/03/2022?")
print(response)