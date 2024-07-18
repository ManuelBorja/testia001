import requests
from getpass import getpass
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_openai.embeddings.base import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_openai.chat_models import ChatOpenAI
from langchain.chains.retrieval_qa.base import RetrievalQA
from dotenv import load_dotenv
import os

load_dotenv('.env')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PATH_ARCHIVOS = os.getenv('PATH_ARCHIVOS')
#os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

def descarga_documentos():
    ##Carga documentos PDF de un URL en una lista de documentos de Langchain
    ml_papers = []
    urls = [
        'https://tc.gob.pe/jurisprudencia/2024/05277-2022-HC.pdf?_gl=1*l1ng4u*_ga*MTM0ODk2MDgzMS4xNzA2NTg5NjUy*_ga_BK92586FH9*MTcyMDM2NTg0NC41LjEuMTcyMDM2NTkwNS42MC4wLjQ2MDkzMTkx',
        'https://tc.gob.pe/jurisprudencia/2024/05240-2022-AA.pdf?_gl=1*l1ng4u*_ga*MTM0ODk2MDgzMS4xNzA2NTg5NjUy*_ga_BK92586FH9*MTcyMDM2NTg0NC41LjEuMTcyMDM2NTkwNS42MC4wLjQ2MDkzMTkx',
        'https://tc.gob.pe/jurisprudencia/2024/05161-2022-HC.pdf?_gl=1*l1ng4u*_ga*MTM0ODk2MDgzMS4xNzA2NTg5NjUy*_ga_BK92586FH9*MTcyMDM2NTg0NC41LjEuMTcyMDM2NTkwNS42MC4wLjQ2MDkzMTkx',
        'https://tc.gob.pe/jurisprudencia/2024/04803-2023-HC.pdf?_gl=1*1y92a03*_ga*MTM0ODk2MDgzMS4xNzA2NTg5NjUy*_ga_BK92586FH9*MTcyMDM2NTg0NC41LjEuMTcyMDM2NTkwNS42MC4wLjQ2MDkzMTkx',
        'https://tc.gob.pe/jurisprudencia/2024/04767-2023-AA.pdf?_gl=1*1y92a03*_ga*MTM0ODk2MDgzMS4xNzA2NTg5NjUy*_ga_BK92586FH9*MTcyMDM2NTg0NC41LjEuMTcyMDM2NTkwNS42MC4wLjQ2MDkzMTkx'
    ]
    
    for i, url in enumerate(urls):
        response = requests.get(url)
        filename = os.path.join("archivos",f'paper{i+1}.pdf')
        with open(filename, 'wb') as f:
            f.write(response.content)
            print(f'Descargado {filename}')

    print("Documentos descargados")

def procesa_descargas():
    ml_papers = []
    
    # Lista todos los archivos en el directorio especificado
    files = os.listdir(PATH_ARCHIVOS)
    # Filtra y cuenta los archivos que terminan con .pdf
    print(files)
    archivos_pdf = [file for file in files if file.lower().endswith('.pdf')]

    for archivo in archivos_pdf:
        filename = os.path.join(PATH_ARCHIVOS,archivo)
        print(filename)
        with open(filename, 'r') as f:
            loader = PyPDFLoader(filename)
            data = loader.load()
            ml_papers.extend(data)

    print("Documentos cargados")
    
    ##Divide todos los documentos en chunks "pequeños" para realizar busquedas
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        length_function=len
        )

    return text_splitter.split_documents(ml_papers)
    ##Guardo los chuncks convertidos en embeddings en una base de datos vectorial (Chroma) para realizar busquedas

def embeddings():
    documents = procesa_descargas()
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings
    )
    print("Embeddings creados")

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )
    print("Documentos almacenados en memoria")
    ##Con las librerias para chat de OpenIA crea la plataforma para la busqueda y proceso de los embeddings
    return retriever

def generar_respuesta(pregunta):
    local_embeddings = embeddings()

    chat = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        model_name='gpt-3.5-turbo',
        temperature=0.0
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=chat,
        chain_type="stuff",
        retriever=local_embeddings
    )

    print("Listo para responder preguntas")
    # Función de Python para generar respuestas

        # Lógica para generar la respuesta basada en la pregunta
    return qa_chain.invoke(pregunta)


