# Agent With Knowledge - Chatbot com Documentos PDF

Este projeto demonstra a implementação de um chatbot inteligente capaz de responder perguntas com base no conteúdo de um documento PDF fornecido pelo usuário. A aplicação utiliza um modelo de linguagem rodando localmente através do Ollama e uma interface web interativa criada com Streamlit.

## Funcionalidades

-   **Interface de Chat Interativa:** Converse com um agente de IA em tempo real.
-   **Base de Conhecimento Dinâmica:** Faça o upload de qualquer arquivo PDF para que sirva como base de conhecimento.
-   **Processamento Local:** Utiliza o Ollama para rodar modelos de linguagem grandes (LLMs) localmente, garantindo privacidade e controle.
-   **Técnica de RAG:** Implementa a técnica de Geração Aumentada por Recuperação (Retrieval-Augmented Generation) para fornecer respostas baseadas em fatos extraídos do documento.
-   **Sistema de Cache:** O processamento pesado do PDF é feito apenas uma vez por arquivo, garantindo respostas rápidas em perguntas subsequentes.

## Estrutura do Projeto

-   `app.py`: O script principal que roda a interface web com Streamlit.
-   `agent_with_knowledge.py`: Contém a classe principal do agente, encapsulando a lógica de RAG.
-   `extract_pdf_to_json.py`: Classe auxiliar para extrair o conteúdo do PDF.
-   `create_db.py`: Script de criação e tratamento da base de dados
-   `requirements.txt`: Lista de dependências Python do projeto.
-   `temp_files/`: Diretório temporário para armazenar os PDFs enviados.
-   `output/`: Diretório temporário para armazenar o conteúdo JSON extraído.
-   `chroma_db/`: Diretório onde o banco de dados vetorial é persistido (opcional, usado pela lógica antiga).

## Pré-requisitos

Para executar este projeto, você precisará ter o seguinte software e bibliotecas instaladas.

### Software

1.  **Python** (versão 3.9 ou superior).
2.  **Ollama:** Ferramenta para rodar LLMs localmente. É **essencial** para o funcionamento do agente.
    -   Faça o download em: [https://ollama.com/](https://ollama.com/)
3.  **Modelos do Ollama:** Após instalar o Ollama, baixe os modelos necessários via terminal:
    ```bash
    # Modelo principal para geração de texto
    ollama pull gemma3:latest 
    
    # Modelo para criação de embeddings
    ollama pull mxbai-embed-large
    ```

### Bibliotecas Python

Todas as dependências estão listadas no arquivo `requirements.txt`. Para instalá-las, ative seu ambiente virtual e execute:

```bash
pip install -r requirements.txt
```

## Como Executar

1.  **Clone ou baixe** esta pasta para o seu computador.
2.  **Instale o Ollama** e baixe os modelos conforme descrito na seção de pré-requisitos.
3.  **Crie e ative um ambiente virtual** Python (recomendado).
4.  **Instale as dependências** com `pip install -r requirements.txt`.
5.  **Execute o Ollama** para que o serviço fique rodando em segundo plano.
6.  **Inicie a aplicação Streamlit** rodando o seguinte comando no seu terminal, na pasta do projeto:
    ```bash
    streamlit run app.py
    ```
7.  Seu navegador abrirá uma nova aba com a interface do chatbot. Faça o upload de um arquivo PDF e comece a conversar!

## Melhorias Futuras

Este projeto serve como uma base sólida. Algumas possíveis melhorias futuras incluem:
-   Implementar um botão para limpar o histórico do chat.
-   Otimizar a extração de texto para tabelas e imagens.
-   Testar diferentes modelos de embedding e de geração para comparar performance e qualidade.
-   Melhorar o tratamento de erros e o feedback para o usuário.