import os
from io import StringIO
import numpy as np
import pandas as pd
import streamlit as st
import nltk
import networkx as nx
nltk.download('punkt') # one time execution
from nltk.tokenize import sent_tokenize
from lexrank import LexRank
from summa.summarizer import summarize as summa_summarizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

## Define summarizer models
# lex_rank
def lex_rank_summarize(article, ratio=0.5):
  sentences = sent_tokenize(article)
  summary = LexRank(sentences).get_summary(sentences,
                             summary_size=int(len(sentences)*0.5), threshold=.1)
  return "\n".join(summary)

# text_rank
def text_rank_summarize(article, ratio):
  return summa_summarizer(article, ratio=ratio)

# text_rank
# def text_rank_summarize(article, ratio=0.5):
  # return summa_summarizer(article, ratio=ratio)

# Define Topline summarizers
def tfidf_summarize(article, ratio=0.5):
  sentences = sent_tokenize(article)
  # get similarity matrix
  sim_mat = cosine_similarity(TfidfVectorizer().fit_transform(sentences))
  scores = nx.pagerank_numpy(nx.from_numpy_array(sim_mat))
  top_ranked = sorted(scores.items(), key=lambda x: x[1], 
                      reverse=True)[:int(len(scores)*ratio)]
  summary = [sentences[i] for i,_ in top_ranked]
  return "\n".join(summary)

# build similarity matrix
def gen_similarity_matrix(sents):
  sim_mat = np.zeros([len(sents), len(sents)])
  for i in range(len(sents)):
    for j in range(len(sents)):
      if i != j:
        sim_mat[i][j] = float(cosine_similarity(sents[i], sents[j]))
  return sim_mat

#helper functions------------------------------------------------------------------
def uploadfile():
    uploaded_file = st.file_uploader("Choose a text file")
    if uploaded_file is not None:
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        return stringio.read()
    else:
        return '<Please upload your file ...>'
#apps------------------------------------------------------------------
def run_text_summarizer():
    language = st.sidebar.selectbox('Newid iaith (Change language):', ['Cymraeg', 'English'])
    with st.expander("ℹ️ - About this app", expanded=False):
        st.write(
            """     
            -   This tool adapts the app from the [Welsh Summarization] (https://github.com/UCREL/welsh-summarization-dataset) project!
            -   It performs simple extractive summarisation with the [TextRank]() alrogithm.
            """
        )

    if language=='Cymraeg':
        st.markdown('## 🌷 Adnodd Creu Crynodebau')
        st.markdown("### Rhowch eich testun isod:")
        option = st.sidebar.radio('Sut ydych chi am fewnbynnu eich testun?', ('Defnyddiwch destun enghreifftiol', 'Rhowch eich testun eich hun', 'Llwythwch ffeil testun i fyny'))
        if option == 'Defnyddiwch destun enghreifftiol':
           example_fname = st.sidebar.selectbox('Select example text:',
                            ['ex_0_Dulyn', 'ex_1_Menter Iaith Môn',
                             'ex_2_Pencampwriaeth', 'ex_3_Paris',
                             'ex_4_Neuadd y Ddinas', 'ex_5_Y_Gofid_Mawr_Covid19'])
           with open(os.path.join('example_texts', example_fname), 'r', encoding='utf8') as example_file:
               example_text = example_file.read()

           input_text = st.text_area('Crynhowch y testun enghreifftiol yn y blwch:', example_text, height=300)
        
        elif option == 'Llwythwch ffeil testun i fyny':
            text = uploadfile()
            input_text = st.text_area("Crynhoi testun wedi'i uwchlwytho:", text, height=300)

        else:
            input_text = st.text_area('Teipiwch neu gludwch eich testun yn y blwch testun', '<Rhowch eich testun...>')

        chosen_ratio = st.sidebar.slider('Dewiswch gymhareb y crynodeb [10% i 50%]:', min_value=10, max_value=50, step=10)/100
        if st.button("Crynhoi👈"):
            if input_text and input_text!='<Rhowch eich testun (Please enter your text...)>':
                summary = text_rank_summarize(input_text, ratio=chosen_ratio)
                if summary:
                    st.write(text_rank_summarize(input_text, ratio=chosen_ratio))
                else:
                    st.write(sent_tokenize(text_rank_summarize(input_text, ratio=0.5))[0])
            else:
                st.write("Rhowch eich testun...(Please enter your text...)")
    else: #English
        st.markdown('## 🌷 Welsh Summary Creator')
        st.markdown("### Enter your text below:")
        option = st.sidebar.radio('How do you want to input your text?', ('Use an example text', 'Paste a copied', 'Upload a text file'))
        if option == 'Use an example text':
           example_fname = st.sidebar.selectbox('Select example text:', ['ex_0_Dulyn', 'ex_1_Menter Iaith Môn', 'ex_2_Pencampwriaeth', 'ex_3_Paris',
           'ex_4_Neuadd y Ddinas', 'ex_5_Y_Gofid_Mawr_Covid19'])
           with open(os.path.join('example_texts', example_fname), 'r', encoding='utf8') as example_file:
               example_text = example_file.read()
               input_text = st.text_area('Summarise the example text in the box:', example_text, height=300)
        elif option == 'Upload a text file':
            text = uploadfile()
            input_text = st.text_area('Summarise uploaded text:', text, height=300)
        else:
            input_text = st.text_area('Type or paste your text into the text box:', '<Please enter your text...>', height=300)

        chosen_ratio = st.sidebar.slider('Select summary ratio [10% to 50%]',  min_value=10, max_value=50, step=10)/100
        if st.button("Summarise👈"):
            if input_text and input_text not in ['<Please enter your text...>','<Please upload your file ...>']:
                summary = text_rank_summarize(input_text, ratio=chosen_ratio)
                if summary:
                    st.write(text_rank_summarize(input_text, ratio=chosen_ratio))
                else:
                    st.write(sent_tokenize(text_rank_summarize(input_text, ratio=0.5))[0])
            else:
              st.write('Please select an example, or paste/upload your text')