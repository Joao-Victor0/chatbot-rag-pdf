import streamlit as st
import os
import re

from agent_with_knowledge import AgentWithKnowledge


def save_uploaded_file(uploaded_file):
    temp_dir = "temp_files"
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, uploaded_file.name)

    with open(file_path, "wb") as file:
        file.write(uploaded_file.getbuffer())

    return file_path


@st.cache_resource
def load_agent_from_file(_uploaded_file):
    st.write(f"Iniciando o processamento do arquivo: {_uploaded_file.name}")
    st.info("Aguarde, este processo pode levar alguns minutos na primeira vez...")
    #st.warning("⚠️ EXECUTANDO A FUNÇÃO DE CACHE `load_agent_from_file`. Isso só deveria aparecer UMA VEZ por arquivo!")

    pdf_path = save_uploaded_file(_uploaded_file) #save the file and takes it path

    #setup agent
    agent = AgentWithKnowledge()
    agent.setup_knowledge_base(pdf_path=pdf_path)

    st.success(f"Arquivo '{_uploaded_file.name}' processado! O agent está pronto!")
    return agent


def main():
    st.title("📄 Agent With Knowledge")
    st.markdown("Faça o upload de um PDF e converse com o agente.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    uploaded_file = st.file_uploader("Escolha um arquivo PDF para começar a conversa", type="pdf")

    if uploaded_file is not None:
        agent = load_agent_from_file(uploaded_file)

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Pergunta algo sobre o documento: "):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("Pensando...."):
                    raw_answer = agent.ask(prompt)
                    clean_answer = re.sub(r"<think>.*?</think>", "", raw_answer, flags=re.DOTALL).strip()
                    st.write(clean_answer)

            st.session_state.messages.append({"role": "assistant", "content": clean_answer})

    else:
        st.info("Aguardando uma pergunta")

if __name__=="__main__":
    main()