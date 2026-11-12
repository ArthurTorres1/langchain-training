import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory


load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

modelo = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.5,
    api_key=api_key
)

perguntas = [
    "Indique uma cidade para passar férias, com foco em atividades de praia.",
    "Quais os melhores restaurantes para famílias com crianças nessa cidade?",
]

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "Se apresente como Sr. Passeios, um especialista em viagens em território brasileiro."),
    ("placeholder", "{historico}"),
    ("human", "{question}")
])

cadeia = prompt_template | modelo | StrOutputParser()

memoria = {}
sessao="alura_agent"

def gerar_historico(sessao):
    if sessao not in memoria:
        memoria[sessao] = InMemoryChatMessageHistory()
    return memoria[sessao]

cadeia_com_memoria = RunnableWithMessageHistory(
    runnable=cadeia,
    get_session_history=gerar_historico,
    input_messages_key="question",
    history_messages_key="historico"
)

for pergunta in perguntas:
    resposta = cadeia_com_memoria.invoke(
        {
            "question": pergunta
        },
        config={"session_id": sessao} 
    )
    print(f"Pergunta: {pergunta}")
    print(f"Resposta: {resposta}\n")



