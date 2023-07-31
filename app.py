import streamlit as st
import time 
import os 
import openai 
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatMessagePromptTemplate, SystemMessagePromptTemplate, AIMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory 
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFacePipeline 
from langchain import PromptTemplate, LLMChain 
from templates import css, bot_template, user_template 
import os 
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.prompts.prompt import PromptTemplate

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY]


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    system_msg = 'You are a helpful assistant. You will be given long documents and will be asked to generate summaries and answer questions. Your responses should be extremely detailed and robust and included specific information. Your responses should be at least one paragraph.'
    llm = ChatOpenAI(temperature=0.2)
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', 
        initial_messages= [{'role': 'system', 'content': system_msg}],
        return_messages=True)
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    if "conversation" not in st.session_state or not st.session_state.conversation:
        st.warning("Please process your PDFs before asking questions.")
        return
    
    if not st.session_state.chat_history:
        system_msg = 'You are a helpful assistant. You will be given long documents and will be asked to generate summaries and answer questions. Your responses should be extremely detailed and robust and included specific information. Your responses should be at least one paragraph.'
        st.session_state.chat_history = [{'role': 'system', 'content': system_msg}]

    if user_question == 'keep_generating_signal':
        # Retrieve the last AI-generated response
        last_ai_response = next((msg.content for msg in reversed(st.session_state.chat_history) if isinstance(msg, AIMessage)), None)

        if last_ai_response:
            # Append " (more elaborate)" to the last AI response to make it longer
            user_question = f'{last_ai_response} (more elaborate)'

    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        elif isinstance(message, AIMessage):
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

    # for i, message in enumerate(st.session_state.chat_history):
    #     if i % 2 == 0:
    #         st.write(user_template.replace(
    #             "{{MSG}}", message.content), unsafe_allow_html=True)
    #     else:
    #         st.write(bot_template.replace(
    #             "{{MSG}}", message.content), unsafe_allow_html=True)
            
    col_regenerate, col_keep_generating = st.columns([1, 1])
    if col_regenerate.button("Regenerate", key = 'regenerate_button_' + str(time.time())):
        if st.session_state.chat_history:
            last_user_question = next((msg.content for msg in reversed(st.session_state.chat_history) if isinstance(msg, HumanMessage)), None)
            if last_user_question:
                handle_userinput(last_user_question)
    
    # if col_keep_generating.button("Keep generating", key='keep_generating_button' + str(time.time())):
    #     if st.session_state.chat_history:
    #         last_user_question = next((msg.content for msg in reversed(st.session_state.chat_history) if isinstance(msg, HumanMessage)), None)
    #         if last_user_question:
    #             st.session_state.last_user_question = last_user_question
    #             handle_userinput('keep_generating_signal')
                



def main():
    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)
                
    
if __name__ == '__main__':
    main()
