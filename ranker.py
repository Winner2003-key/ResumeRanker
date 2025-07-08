from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader   
from langchain.vectorstores import FAISS
from langchain_openai import AzureChatOpenAI
import os
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader, PyPDFLoader
import tempfile
from langchain_openai import AzureChatOpenAI
from langchain.embeddings import AzureOpenAIEmbeddings
import streamlit as st
from PyPDF2 import PdfReader
from langchain.prompts import PromptTemplate


load_dotenv()


embedding_model = AzureOpenAIEmbeddings(
            azure_endpoint=os.getenv("EMBEDDING_API_BASE"),
            openai_api_key=os.getenv("EMBEDDING_API_KEY"),
            deployment=os.getenv("EMBEDDING_DEPLOYMENT_NAME"),
            chunk_size=10,
        )
        
llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_API_BASE"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    temperature=0.7,
    max_tokens=800,
    model_name="gpt-4o"
)

#Not being used
def load_file(file):
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".pdf") as tmp_file:
        tmp_file.write(file.getvalue())
        tmp_file_path = tmp_file.name
    
    loader = PyPDFLoader(file)
    document = loader.load()
    




st.title("AI Resume Ranker")
st.markdown("Uploads")

# --- Upload JD ---
jd_file = st.file_uploader("Upload Job Description (PDF or Text)", type=["pdf", "txt"])
jd_text = ""


if jd_file:
    if jd_file.type == "application/pdf":
        reader = PdfReader(jd_file)
        jd_text = "\n".join([page.extract_text() for page in reader.pages])
    else:
        jd_text = jd_file.read().decode("utf-8")
        
# --- Upload Resumes ---
resume_files = st.file_uploader("Upload up to 10 and more Resumes (PDF)", type="pdf", accept_multiple_files=True)
resume_texts = {}

results = []

if st.button("Rank Resumes"):
    if jd_text and resume_files:
        st.info("Processing resumes and ranking...")

        # Extract text from resumes
        for file in resume_files:
            reader = PdfReader(file)
            resume_text = "\n".join([page.extract_text() for page in reader.pages])
            resume_texts[file.name] = resume_text

        with st.spinner("Ranking resumes..."):
            for filename, text in resume_texts.items():
                prompt = PromptTemplate(
                    input_variables=["job_description", "resume_text"],
                    template="""
                        You are a hiring assistant. Compare the resume against the job description.

                        Job Description:
                        {job_description}

                        Resume:
                        {resume_text}

                        Rate this resume out of 100 based on how well it fits the job description. Be strict and unbiased.

                        Respond ONLY with a number (e.g., 78).
                    """
                )

                formatted_prompt = prompt.format(
                    job_description=jd_text[:3000],
                    resume_text=text[:3000]
                )

                try:
                    score_response = llm.predict(formatted_prompt)
                    score = int(''.join(filter(str.isdigit, score_response)))
                    results.append((filename, score))
                except Exception as e:
                    results.append((filename, 0))

        # Sort by score
        results.sort(key=lambda x: x[1], reverse=True)

        # Display Results
        st.subheader("🏆 Ranked Resumes")
        for name, score in results:
            st.markdown(f"*{name}*: {score}/100")






