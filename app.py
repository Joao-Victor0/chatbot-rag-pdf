import streamlit as st
from agent_with_knowledge import AgentWithKnowledge

def main():
    agent = AgentWithKnowledge()

    #Estilização
    st.header("📄 ChatBot Inteligente", divider=True)
    st.markdown("Converse com o agente sobre os documentos")

    #Sessão do Usuário
    if "messages" not in st.session_state:
        st.session_state.messages = []

    #Configurando o espaço de mensagens
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    #Escrevendo a mensagem do usuário e do agente nos campos apropriados
    if question:= st.chat_input("Digite aqui sua pergunta"):
        st.session_state.messages.append({"role": "user", "content": question})

        with st.chat_message("user"): #abre o campo do usuário
            st.markdown(question) #coloca a mensagem do usuário na tela

        with st.chat_message("assistant"): #abre o campo do agente
            with st.spinner("Pensando..."): #cria um efeito enquanto a resposta é carregada
                response = agent.ask(query=question)
                st.write(response) #coloca a resposta do agente na tela

        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__=="__main__":
    main()