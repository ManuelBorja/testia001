import requests
from getpass import getpass
import os

OPENAI_API_KEY = getpass('API Key de OPEN AI: ')
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY


##Carga documentos PDF de un URL en una lista de documentos de Langchain
from langchain_community.document_loaders.pdf import PyPDFLoader

urls = [
    'https://tc.gob.pe/jurisprudencia/2024/05277-2022-HC.pdf?_gl=1*l1ng4u*_ga*MTM0ODk2MDgzMS4xNzA2NTg5NjUy*_ga_BK92586FH9*MTcyMDM2NTg0NC41LjEuMTcyMDM2NTkwNS42MC4wLjQ2MDkzMTkx',
    'https://tc.gob.pe/jurisprudencia/2024/05240-2022-AA.pdf?_gl=1*l1ng4u*_ga*MTM0ODk2MDgzMS4xNzA2NTg5NjUy*_ga_BK92586FH9*MTcyMDM2NTg0NC41LjEuMTcyMDM2NTkwNS42MC4wLjQ2MDkzMTkx',
    'https://tc.gob.pe/jurisprudencia/2024/05161-2022-HC.pdf?_gl=1*l1ng4u*_ga*MTM0ODk2MDgzMS4xNzA2NTg5NjUy*_ga_BK92586FH9*MTcyMDM2NTg0NC41LjEuMTcyMDM2NTkwNS42MC4wLjQ2MDkzMTkx',
    'https://tc.gob.pe/jurisprudencia/2024/04803-2023-HC.pdf?_gl=1*1y92a03*_ga*MTM0ODk2MDgzMS4xNzA2NTg5NjUy*_ga_BK92586FH9*MTcyMDM2NTg0NC41LjEuMTcyMDM2NTkwNS42MC4wLjQ2MDkzMTkx',
    'https://tc.gob.pe/jurisprudencia/2024/04767-2023-AA.pdf?_gl=1*1y92a03*_ga*MTM0ODk2MDgzMS4xNzA2NTg5NjUy*_ga_BK92586FH9*MTcyMDM2NTg0NC41LjEuMTcyMDM2NTkwNS42MC4wLjQ2MDkzMTkx'
]

ml_papers = []

for i, url in enumerate(urls):
    response = requests.get(url)
    filename = f'paper{i+1}.pdf'
    with open(filename, 'wb') as f:
        f.write(response.content)
        print(f'Descargado {filename}')

        loader = PyPDFLoader(filename)
        data = loader.load()
        ml_papers.extend(data)


##Divide todos los documentos en chunks "pequeños" para realizar busquedas

from langchain_text_splitters.character import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=200,
    length_function=len
    )

documents = text_splitter.split_documents(ml_papers)


##Guardo los chuncks convertidos en embeddings en una base de datos vectorial (Chroma) para realizar busquedas
from langchain_openai.embeddings.base import OpenAIEmbeddings
from langchain_chroma import Chroma


embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

##Con las librerias para chat de OpenIA crea la plataforma para la busqueda y proceso de los embeddings
from langchain_openai.chat_models import ChatOpenAI
from langchain.chains.retrieval_qa.base import RetrievalQA

chat = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model_name='gpt-3.5-turbo',
    temperature=0.0
)

qa_chain = RetrievalQA.from_chain_type(
    llm=chat,
    chain_type="stuff",
    retriever=retriever
)


##Ejecuta la cadena con consultas que muestran los resultados 
print("=============================")
query = "cual es el fallo del tribunal en el caso de LUIS ALBERTO PALACIOS PAYTAN?"
print(qa_chain.invoke(query))
print("=============================")
query = "cual es el fallo del tribunal en la resolución 05240-2022-AA?"
print(qa_chain.invoke(query))
