import os

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv



load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

numero_dias = 7
numero_criancas = 2
atividade = "praia"

modelo_de_prompt = PromptTemplate(
    template="""
    Crie um um roteiro de viagem de {dias} dias,
    para uma família com {numero_criancas} crianças,
    com foco em atividades de {atividade}.
    """
)

prompt = modelo_de_prompt.format(
    dias=numero_dias,
    numero_criancas=numero_criancas,
    atividade=atividade
)

print("Prompt: \n", prompt)

modelo = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.5,
    api_key=api_key
)

resposta = modelo.invoke(prompt)
print(resposta.content)