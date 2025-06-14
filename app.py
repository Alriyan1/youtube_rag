import streamlit as st
#from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from ulits.helperfunction import create_db_from_youtube_video_url,get_response_from_query
import textwrap
from langchain_community.embeddings import HuggingFaceEmbeddings

st.title('Youtube Video info.....')
url = st.text_input('Enter a YT video link')


embeddings = HuggingFaceEmbeddings()

if url:

    db=create_db_from_youtube_video_url(url,embeddings)
    query='what is this video about?'
    response,docs=get_response_from_query(db,query)
    st.write(textwrap.fill(response,width=70))


