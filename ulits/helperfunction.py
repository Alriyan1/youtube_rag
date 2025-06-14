from langchain_groq import ChatGroq
from langchain.vectorstores import FAISS
from langchain.chains import LLMChain
from langchain.prompts.chat import ChatPromptTemplate,SystemMessagePromptTemplate,HumanMessagePromptTemplate
from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import sentence_transformer

def create_db_from_youtube_video_url(video_url,embeddings):
  loader=YoutubeLoader.from_youtube_url(video_url)
  transcript=loader.load()

  text_splitter=RecursiveCharacterTextSplitter(chunk_size=2000,chunk_overlap=100)
  docs=text_splitter.split_documents(transcript)
  db=FAISS.from_documents(docs,embeddings)
  return db


def get_response_from_query(db,query,k=4):
  docs=db.similarity_search(query,k=k)
  docs_page_content = ' '.join([d.page_content for d in docs])
  chat = ChatGroq(api_key = "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", model = "llama-3.3-70b-versatile", temperature=0)

  template = """You are a helpful assistant that that can answer questions about youtube videos
        based on the video's transcript: {docs}

        Only use the factual information from the transcript to answer the question.

        If you feel like you don't have enough information to answer the question, say "I don't know"."""

  system_message_prompt = SystemMessagePromptTemplate.from_template(template)
  human_template = "Answer the following question: {question}"

  human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

  chat_prompt = ChatPromptTemplate.from_messages(
      [system_message_prompt, human_message_prompt]
  )

  chain = LLMChain(llm = chat, prompt = chat_prompt)

  response = chain.run(question = query, docs = docs_page_content)

  response = response.replace("\n", "")

  return response, docs