import os

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain.globals import set_debug
from pydantic import Field, BaseModel
from dotenv import load_dotenv

set_debug(True)

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

class Destino(BaseModel):
    cidade: str = Field(description="Nome da cidade sugerida")
    motivo: str = Field(description="Motivo pelo qual é interessante visitar essa cidade")

class Restaurantes(BaseModel):
    cidade: str = Field(description="Nome da cidade sugerida")
    restaurantes:str = Field(description="Restaurantes recomendados na cidade")

parseador_destino = JsonOutputParser(pydantic_object=Destino)
parseador_restaurantes = JsonOutputParser(pydantic_object=Restaurantes)

prompt_cidade = PromptTemplate(
    template="""
    Sugira uma cidade dado o meu interesse por {interesse}.
    {formato_de_saida}
    """,
    input_variables=["interesse"],
    partial_variables={"formato_de_saida": parseador_destino.get_format_instructions()},
)

prompt_restaurantes = PromptTemplate(
    template="""
    Sugira restaurantes em {cidade}.
    {formato_de_saida}
    """,
    input_variables=["cidade"],
    partial_variables={"formato_de_saida": parseador_restaurantes.get_format_instructions()},
)

prompt_cultural = PromptTemplate(
    template="""
    Sugira atrações culturais em {cidade}.
    """
)

modelo = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.5,
    api_key=api_key
)

cadeia1 = prompt_cidade | modelo | parseador_destino
cadeia2 = prompt_restaurantes | modelo | parseador_restaurantes
cadeia3 = prompt_cultural | modelo | StrOutputParser()

cadeia = (cadeia1 | cadeia2 | cadeia3)

resposta = cadeia.invoke({"interesse": "montanhas"})
print(resposta)