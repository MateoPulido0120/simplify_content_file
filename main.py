import streamlit as st
import io
import time
import json
from contextlib import redirect_stdout
from pypdf import PdfReader
import google.generativeai as genai
from llama_index.core import VectorStoreIndex
from llama_index.core.objects import ObjectIndex, SimpleObjectNodeMapping
from llama_index.llms.gemini import Gemini
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import QueryEngineTool
from llama_index.core.memory import ChatMemoryBuffer
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from tenacity import retry, stop_after_attempt, wait_exponential


def initialize_session():
    if "model_llm" not in st.session_state:
        genai.configure(api_key=st.secrets["API_GEMINI"])
        generation_config=genai.GenerationConfig(response_mime_type="application/json", candidate_count=1, temperature=0, max_output_tokens=1000)
        safety_settings={
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE
            }
 
        st.session_state['model_llm'] = genai.GenerativeModel("gemini-1.5-flash", generation_config=generation_config, safety_settings=safety_settings) 

    if "llm" not in st.session_state:
        # st.session_state['llm'] = Gemini(model="models/gemini-1.5-flash", api_key=st.secrets["API_GEMINI"], max_tokens=1000, temperature=0)
        st.session_state['llm'] = OpenAI(model="gpt-3.5-turbo", api_key=st.secrets["API_OPENAI"])
    
    if "embed_model" not in st.session_state:
        # st.session_state['embed_model'] = GeminiEmbedding(model_name="models/embedding-001", api_key=st.secrets["API_GEMINI"])
        st.session_state['embed_model'] = OpenAIEmbedding(model="text-embedding-3-large", api_key=st.secrets["API_OPENAI"])

    if "memory" not in st.session_state:
        st.session_state['memory'] = ChatMemoryBuffer.from_defaults(llm=st.session_state['llm'], token_limit=8000)

    if 'resume_global_content' not in st.session_state:
        st.session_state['resume_global_content'] = None

    if 'object_retriever' not in st.session_state:
        st.session_state['object_retriever'] = None

    if 'expert_agent' not in st.session_state:
        st.session_state['expert_agent'] = None

    if 'apprentice_agent' not in st.session_state:
        st.session_state['apprentice_agent'] = None

    if 'content_conversation' not in st.session_state:
        st.session_state['content_conversation'] = None

    if 'response_formated' not in st.session_state:
        st.session_state['response_formated'] = None

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def generate_content_pages(text):
    # return llm.complete(f"Extract relevant content from the text and at least 3 key points: {text}").text
    response = st.session_state['model_llm'].generate_content(
        f"""Extract relevant content from the text and at least 3 key points: {text}
        using JSON schema:
        {{
            "content": str,
            "mainly_points": list,
        }}:
        """).text
    
    response = json.loads(response)
    return response

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def generate_global_content(text):
    response = st.session_state['model_llm'].generate_content(
        f"""generates a global summary of all content (no more than 50 words): {text}
        using JSON schema:
        {{
            "content": str,
        }}:
        """).text
    
    response = json.loads(response)
    return response["content"]

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def generate_response(query, context):
    response = st.session_state['model_llm'].generate_content(
        f"""Answer the question : {query}
        using the context: {context}
        using JSON schema:
        {{
            "response": str,
        }}:
        """).text
    
    response = json.loads(response)
    return response["response"]

def create_object_retriever(list_objects):
    # object-node mapping and nodes
    obj_node_mapping = SimpleObjectNodeMapping.from_objects(list_objects)
    nodes = obj_node_mapping.to_nodes(list_objects)

    # object index
    object_index = ObjectIndex(
        index=VectorStoreIndex(nodes=nodes, embed_model=st.session_state['embed_model']),
        object_node_mapping=obj_node_mapping,
    )

    return object_index.as_retriever(similarity_top_k=2)

def process_data(bytes_data):
    pdf_bytes = io.BytesIO(bytes_data)
    reader = PdfReader(pdf_bytes)
    
    pages_content = []
    c = 0
    for page in reader.pages:
        c+=1
        pages_content.append(generate_content_pages(page.extract_text()))
        time.sleep(2)

        # if c > 5:
        #     break
    
    global_content = "\n".join([element['content'] for element in pages_content])
    resume_global_content = generate_global_content(global_content)

    object_retriever = create_object_retriever(pages_content)

    return resume_global_content, object_retriever

def parameterize_expert_agent(resume_global_content, object_retriever):
        def extract_relevant_content(question: str) -> list:
            """Extract relevant content from context document"""
            response_reliable = object_retriever.retrieve(question)
            return generate_response(question, response_reliable)
    
        extract_relevant_content_tool = FunctionTool.from_defaults(fn=extract_relevant_content)

        prompt_context = f"""
        - You are an expert in {resume_global_content}. 
        - Your goal is to answer all the questions asked by the user until the user has complete clarity on the topic.
        - Use the tool 'extract_relevant_content_tool' to extract relevant content to response the questions.
        """

        return ReActAgent.from_tools([extract_relevant_content_tool], llm=st.session_state['llm'], context=prompt_context, verbose=True, 
                                     max_iterations=40)

def parameterize_apprentice_agent(expert_agent, resume_global_content):
    tool_expert_agent = QueryEngineTool.from_defaults(
        query_engine=expert_agent,
        name = f"tool_expert_agent",
        description = f"The tool tool_expert_agent is useful for answering questions about specific topics"
    )
    

    system_prompt = f"""
    - You are an agent with a great interest in learning more about {resume_global_content}.
    - Your goal is to ask questions that allow you to learn more and more about the entire topic.
    - You must ask questions until you have complete clarity on the topic.
    - You can use the tool 'tool_expert_agent' to answer your questinons.
    """


    return ReActAgent.from_tools(
        [tool_expert_agent],
        llm=st.session_state['llm'],
        memory=st.session_state['memory'],
        context=system_prompt,
        verbose=True,
        max_iterations=40,
    )

def generate_output_content(content_conversation):

    llm_pro = OpenAI(model="gpt-4o-mini", api_key=st.secrets["API_OPENAI"])

    response = llm_pro.complete(
        f"""From the entire text corresponding to a conversation between an expert on the subject and an apprentice: {content_conversation}.

        Generate the following content format for students:

        - Introductory overview of the topic.
        - Detailed explanation of the key points of the topic.
        - Practical examples and solutions.
        - Summary and next steps for students.

        Use markdown format:""")
    
    return response.text

if __name__ == "__main__":

    initialize_session()

    st.title("Content generator for simplified learning")
    st.write("""This project aims to process content (currently PDF) from plain text files, extracting important features from all the content. 
             Its simplified or summarized generation is done from the conversation obtained between two ReAct (Reasoning and Actuation) agents, 
             one of them with the role of teacher or expert on the subject and the other agent with the role of learner, so that Among them, 
             questions and answers are asked about the specific topic that the document deals with and to be able to have a closer simulation of the true relevant content of the file.
             """)
    
    st.image("static/image_architecture.PNG")

    uploaded_file = st.file_uploader("Choose a file", accept_multiple_files=False, type="PDF")
    if uploaded_file is not None:
        if st.session_state['resume_global_content'] is None and st.session_state['object_retriever'] is None:
            bytes_data = uploaded_file.getvalue()
            with st.spinner(text="Processing file..."):
                st.session_state['resume_global_content'], st.session_state['object_retriever']  = process_data(bytes_data)
                st.toast('Processed!', icon='✅')

    if st.session_state['resume_global_content'] is not None and st.session_state['object_retriever']:
        if st.session_state['expert_agent'] is None and st.session_state['apprentice_agent'] is None:
            st.session_state['expert_agent'] = parameterize_expert_agent(st.session_state['resume_global_content'], st.session_state['object_retriever'])
            st.session_state['apprentice_agent'] = parameterize_apprentice_agent(st.session_state['expert_agent'], st.session_state['resume_global_content'])

    if st.session_state['expert_agent'] is not None and st.session_state['apprentice_agent'] is not None:
        with st.spinner(text="Processing internal conversation between agents..."):
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                response = st.session_state['apprentice_agent'].chat("hi, i want to learn")

            st.toast('Processed!', icon='✅')

        with st.spinner(text="Generating required format..."):
            st.session_state['content_conversation'] = buffer.getvalue()
            st.session_state['response_formated'] = generate_output_content(st.session_state['content_conversation'])
            st.toast('Processed!', icon='✅')

    if st.session_state['response_formated'] is not None:
        st.subheader("Result:")
        st.write(st.session_state['response_formated'])

    if st.session_state['content_conversation'] is not None:
        content_bytes = st.session_state['content_conversation'].encode('utf-8')

        st.download_button(
            label="download agents conversation file",
            data=content_bytes,
            file_name="agents conversation.txt",
            mime="text/plain"
        )