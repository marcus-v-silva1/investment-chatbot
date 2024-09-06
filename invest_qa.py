import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import load_dotenv
from langchain_community.document_loaders import CSVLoader

load_dotenv()

load = CSVLoader(file_path="knowInvest_basa.csv")
data = load.load()

embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(data, embeddings)

def calc_similar(question):
  similar_results = db.similarity_search(question, k=3)
  return [doc.page_content for doc in similar_results]

llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

template = """ 
Você é um analista de investimentos da XP, renomada empresa de assessoria 
financeira e investimentos no Brasil. 
Seu objetivo é fornecer informações claras, 
detalhadas e confiáveis aos clientes sobre suas dúvidas e consultas financeiras. 
Você deve responder de forma profissional, cortês e utilizando termos técnicos do 
mercado financeiro de forma compreensível para todos os níveis de conhecimento.
Siga todas as regras abaixo:
1/ você deve se comportar como um analista de investimentos.
2/ você deve respeitar as regras de negócios e as regras da XP.
3/ suas respostas devem ser similates ou até parecidas com as perguntas da lista fornecida abaixo.
4/ Perguntas fora da lista devem ser respondidas apenas se for sobre investimentos, caso o contrario responda "não fui programado para responder sobre isso".

Vou lhe passar algumas dúvidas dos clientes:
{message}

Aqui é uma lista de perguntas e respostas criados pela XP.
Essa lista servirá como base para que você possa responder as dúvidas dos clientes.
{base}

escreva a melhor resposta para que eu envie para esse cliente e sanar suas dúvidas:
"""

prompt = PromptTemplate(
  input_variables=["message" "base"],
  template=template
)

chain = LLMChain(llm=llm, prompt=prompt)
def generate_answer(message):
  base = calc_similar(message)
  response = chain.run(message=message, base=base)
  return response

def main():
  st.set_page_config(
      page_title="Assessor de investimentos", page_icon="declaracoes-financeiras.png") 
  st.header("Qual é sua dúvida sobre investimentos? 🪙")

  message = st.text_input("Digite sua dúvida sobre investimentos aqui:")

  if message:
    st.write("Gerando uma resposta para a sua dúvida...")
    result = generate_answer(message)
    st.info(result)

if __name__ == "__main__":
    main()