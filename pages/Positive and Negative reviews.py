import streamlit as st
import base64
from PIL import Image
from labels import MESSAGES
import os
import string
import random
import pandas as pd
import numpy as np

from collections import Counter
import spacy
from spacy.tokens import Doc
from spacy.vocab import Vocab
import nltk

import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from PIL import Image
from textblob import TextBlob
from nltk import word_tokenize, sent_tokenize, ngrams
from wordcloud import WordCloud, ImageColorGenerator
from nltk.corpus import stopwords
from labels import MESSAGES
from summarizer_labels import SUM_MESSAGES
from summa.summarizer import summarize as summa_summarizer
from langdetect import detect
nltk.download('punkt') # one time execution
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

from pathlib import Path
from typing import List
##word association
import networkx as nx
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import base64

import circlify ###### pip install circlify
import plotly.express as px #### pip install plotly.express
#from pyvis.network import Network
import streamlit.components.v1 as components


import scattertext as tt
import spacy
from pprint import pprint
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode
@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(png_file):
    with open(png_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


def build_markup_for_logo(
    png_file,
    background_position="50% 10%",
    margin_top="10%",
    image_width="60%",
    image_height="",
):
    binary_string = get_base64_of_bin_file(png_file)
    return """
            <style>
                [data-testid="stSidebarNav"] {
                    background-image: url("data:image/png;base64,%s");
                    background-repeat: no-repeat;
                    background-position: %s;
                    margin-top: %s;
                    background-size: %s %s;
                }
            </style>
            """ % (
        binary_string,
        background_position,
        margin_top,
        image_width,
        image_height,
    )


def add_logo(png_file):
    logo_markup = build_markup_for_logo(png_file)
    st.markdown(
        logo_markup,
        unsafe_allow_html=True,
    )



# ----------------
st.set_page_config(
     page_title='Welsh Free Text Tool',
     page_icon='🌐',
     layout="wide",
     initial_sidebar_state="expanded",
     menu_items={
         'Get Help': "https://ucrel.lancs.ac.uk/freetxt/",
         'Report a bug': "https://github.com/UCREL/welsh-freetxt-app/issues",
         'About': '''## The FreeTxt tool supports bilingual (English and Welsh) free text data analysis of surveys and questionnaire responses'''
     }
 )

st.markdown("# Positive and Negative reviews")

        
add_logo("img/FreeTxt_logo.png") 
st.write("---")
st.write("This tool performs sentiment analyses on your reviews, displaying the results in a pie chart")


# Update with the Welsh stopwords (source: https://github.com/techiaith/ataleiriau)
en_stopwords = list(stopwords.words('english'))
cy_stopwords = open('welsh_stopwords.txt', 'r', encoding='iso-8859-1').read().split('\n') # replaced 'utf8' with 'iso-8859-1'
STOPWORDS = set(en_stopwords + cy_stopwords)
PUNCS = '''!→()-[]{};:'"\,<>./?@#$%^&*_~'''
pd.set_option('display.max_colwidth',None)

lang='en'
EXAMPLES_DIR = 'example_texts_pub'



# reading example and uploaded files
def read_file(fname, file_source):
    file_name = fname if file_source=='example' else fname.name
    if file_name.endswith('.txt'):
        data = open(fname, 'r', encoding='cp1252').read().split('\n') if file_source=='example' else fname.read().decode('utf8').split('\n')
        data = pd.DataFrame.from_dict({i+1: data[i] for i in range(len(data))}, orient='index', columns = ['Reviews'])
        
    elif file_name.endswith(('.xls','.xlsx')):
        data = pd.read_excel(pd.ExcelFile(fname)) if file_source=='example' else pd.read_excel(fname)

    elif file_name.endswith('.tsv'):
        data = pd.read_csv(fname, sep='\t', encoding='cp1252') if file_source=='example' else pd.read_csv(fname, sep='\t', encoding='cp1252')
    else:
        return False, st.error(f"""**FileFormatError:** Unrecognised file format. Please ensure your file name has the extension `.txt`, `.xlsx`, `.xls`, `.tsv`.""", icon="🚨")
    return True, data

def get_data(file_source='example'):
    try:
        if file_source=='example':
            example_files = sorted([f for f in os.listdir(EXAMPLES_DIR) if f.startswith('Reviews')])
            fnames = st.sidebar.multiselect('Select example data file(s)', example_files, example_files[0])
            if fnames:
                return True, {fname:read_file(os.path.join(EXAMPLES_DIR, fname), file_source) for fname in fnames}
            else:
                return False, st.info('''**NoFileSelected:** Please select at least one file from the sidebar list.''', icon="ℹ️")
        
        elif file_source=='uploaded': # Todo: Consider a maximum number of files for memory management. 
            uploaded_files = st.sidebar.file_uploader("Upload your data file(s)", accept_multiple_files=True, type=['txt','tsv','xlsx', 'xls'])
            if uploaded_files:
                return True, {uploaded_file.name:read_file(uploaded_file, file_source) for uploaded_file in uploaded_files}
            else:
                return False, st.info('''**NoFileUploaded:** Please upload files with the upload button or by dragging the file into the upload area. Acceptable file formats include `.txt`, `.xlsx`, `.xls`, `.tsv`.''', icon="ℹ️")
        else:
            return False, st.error(f'''**UnexpectedFileError:** Some or all of your files may be empty or invalid. Acceptable file formats include `.txt`, `.xlsx`, `.xls`, `.tsv`.''', icon="🚨")
    except Exception as err:
        return False, st.error(f'''**UnexpectedFileError:** {err} Some or all of your files may be empty or invalid. Acceptable file formats include `.txt`, `.xlsx`, `.xls`, `.tsv`.''', icon="🚨")


def select_columns(data, key):
    layout = st.columns([7, 0.2, 2, 0.2, 2, 0.2, 3, 0.2, 3])
    selected_columns = layout[0].multiselect('Select column(s) below to analyse', data.columns, help='Select columns you are interested in with this selection box', key= f"{key}_cols_multiselect")
    start_row=0
    if selected_columns: start_row = layout[2].number_input('Choose start row:', value=0, min_value=0, max_value=5)
    
    if len(selected_columns)>=2 and layout[4].checkbox('Filter rows?'):
        filter_column = layout[6].selectbox('Select filter column', selected_columns)
        if filter_column: 
            filter_key = layout[8].selectbox('Select filter key', set(data[filter_column]))
            data = data[selected_columns][start_row:].dropna(how='all')
            return data.loc[data[filter_column] == filter_key].drop_duplicates()
    else:
        return data[selected_columns][start_row:].dropna(how='all').drop_duplicates()


import polyglot
from polyglot.text import Text
#text = Text("The movie was really good.")
from polyglot.downloader import downloader
downloader.download("TASK:sentiment2")
#st.write("{:<16}{}".format("Word", "Polarity")+"\n"+"-"*30)
#for w in text.words:
   # st.write("{:<16}{:>2}".format(w, w.polarity))
  

# --------------------Sentiments-----------------------

#---Polarity score

def get_sentiment(polarity):
    return 'Very Positive' if polarity >= 0.5 else 'Positive' if (
        0.5 > polarity > 0.0) else 'Negative' if (0.0 > polarity >= -0.5
        ) else 'Very Negative' if -0.5 > polarity else 'Neutral'

def get_text_sentiments(reviews):
    results = []
    for review in reviews:
        text = Text(review)
        polarity = sum([w.polarity for w in text.words if isinstance(w.polarity, float)])
        num_words = len([w for w in text.words if isinstance(w.polarity, float)])
        avg_polarity = polarity / num_words if num_words > 0 else 0
        sentiment = get_sentiment(avg_polarity)
        results.append((review, avg_polarity, sentiment))
    return results

###########Bert
from transformers import AutoTokenizer, AutoModelForSequenceClassification
st.write("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")

st.write("Loading model...")
model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")

print("Model loaded.")

text = "This is a positive sentence."
inputs = tokenizer.encode_plus(
    text,
    add_special_tokens=True,
    return_attention_mask=True,
    return_tensors="pt"
)
outputs = model(**inputs)
scores = outputs.logits.softmax(dim=1).detach().numpy()[0]
sentiment_labels = ['negative', 'somewhat negative', 'neutral', 'somewhat positive', 'positive']
sentiment_index = scores.argmax()
sentiment_label = sentiment_labels[sentiment_index]

st.write(f"The sentiment of '{text}' is '{sentiment_label}' with a score of {scores[sentiment_index]:.2f}.")

   

  
# ---------------------
def plot_sentiments(data, fine_grained=True):
  fig, ax = plt.subplots(figsize=(5,5))
  size = 0.7 
  cmap = plt.get_cmap("tab20c")

  if not fine_grained:
    new_pos = tuple(map(lambda x, y: x + y, data[0], data[1]))
    new_neg = tuple(map(lambda x, y: x + y, data[3], data[4]))
    data = new_pos, data[2], new_neg
    color_code = [8, 3, 4]
    labels = ["Positive", "Neutral", "Negative"]
  else:
    color_code =  [8, 10, 3, 5, 4]
    labels = ["Very Positive", "Positive", "Neutral", "Negative", "Very Negative"]

  vals = np.array(data)
  outer_colors =  cmap(np.array(color_code)) # cmap(np.arange(5))
  # inner_colors = cmap(np.arange(10)) # cmap(np.array([1, 2, 3, 4, 5, 6, 7,8, 9, 10]))

  wedges, texts, autotexts = ax.pie(vals.sum(axis=1), radius=1,
        autopct=lambda pct: plotfunc(pct, data),
        colors=outer_colors, wedgeprops=dict(width=size, edgecolor='w'),
        pctdistance=0.60, textprops=dict(color="w", weight="bold", size=8))

  # ax.set_title("Sentiment Chart")
  
  ax.legend(wedges, labels, title="Sentiment classes", title_fontsize='small', loc="center left", fontsize=8,
            bbox_to_anchor=(1, 0, 0.5, 1))

  st.set_option('deprecation.showPyplotGlobalUse', False)
  st.pyplot()


st.markdown('''🎲 Sentiment Analyzer''')
option = st.sidebar.radio(MESSAGES[lang][0], (MESSAGES[lang][1], MESSAGES[lang][2]))
if option == MESSAGES[lang][1]: input_data = get_data()
elif option == MESSAGES[lang][2]: input_data = get_data(file_source='uploaded')
else: pass
status, data = input_data


    
if status:
        option = st.radio('How do you want to categorize the sentiments?', ('3 Class Sentiments (Positive, Neutral, Negative)', '5 Class Sentiments (Very Positive, Positive, Neutral, Negative, Very Negative)'))
        # With tabbed multiselect
        filenames = list(data.keys())
        #tab_titles= [f"File-{i+1}" for i in range(len(filenames))]
        #tabs = st.tabs(tab_titles)
        for i in range(len(filenames)):
            #with tabs[i]:
                _, df = data[filenames[i]]
                df = select_columns(df, key=i).astype(str)
                if df.empty:
                    st.info('''**NoColumnSelected 🤨**: Please select one or more columns to analyse.''', icon="ℹ️")
                else:
                    
                    tab1, tab2 = st.tabs(["📈 Meaning analysis",'💬 Keyword scatter'])
                    with tab1:
                        
                        input_text = '/n'.join(['/n'.join([str(t) for t in list(df[col]) if str(t) not in STOPWORDS and str(t) not in PUNCS]) for col in df])
                        
                       # text = get_text_sentiments(input_text)
                        #if option == '3 Class Sentiments  (Positive, Neutral, Negative)':
                         #  plot_sentiments(text[1], fine_grained=False)
                        #else:
                         #  plot_sentiments(text[1])
                       # num_examples = st.slider('Number of example [5 to 20%]',  min_value=5, max_value=20, step=5, key=i)
                       # df = pd.DataFrame(text[0], columns =['Review','Polarity', 'Sentiment', 'Subjectivity', 'Category'])
                       # df = df[['Review','Polarity', 'Sentiment']]
                       # df.index = np.arange(1, len(df) + 1)
                    with tab2:
                          #### interactive dataframe
                         gb = GridOptionsBuilder.from_dataframe(df)
                         gb.configure_pagination(paginationAutoPageSize=True) #Add pagination
                         gb.configure_side_bar() #Add a sidebar
                         gb.configure_selection('multiple', use_checkbox=True, groupSelectsChildren="Group checkbox select children") #Enable multi-row selection
                         gridOptions = gb.build()

                         grid_response = AgGrid(
                              df,
                              gridOptions=gridOptions,
                               data_return_mode='AS_INPUT', 
                            update_mode='MODEL_CHANGED', 
                             fit_columns_on_grid_load=False,
    
                                  enable_enterprise_modules=True,
                             height=350, 
                              width='100%',
                              reload_data=True
                                                )
                         data = grid_response['data']
                         selected = grid_response['selected_rows'] 
                         df = pd.DataFrame(selected) #Pass the selected rows to a new dataframe df
                        
                         #AgGrid(df.head(num_examples),use_container_width=True)
                         HtmlFile = open("Visualization.html", 'r', encoding='utf-8')
                         source_code = HtmlFile.read() 
                         print(source_code)
                         components.html(source_code,height = 800)
                            
                            
                         #convention_df = tt.SampleCorpora.ConventionData2012.get_data()  
                         #convention_df.iloc[0]
                         nlp = spacy.load('en_core_web_sm-3.2.0')  
                         nlp.max_length = 9000000
                         #corpus = tt.CorpusFromPandas(convention_df, 
                          #   category_col='party', 
                           #  text_col='text',
                            # nlp=nlp).build()
                         #st.write(corpus)
                         #term_freq_df = corpus.get_term_freq_df()
                         #term_freq_df['positive Score'] = corpus.get_scaled_f_scores('democrat')
                         #term_freq_df['negative Score'] = corpus.get_scaled_f_scores('republican')

                         #html = tt.produce_scattertext_explorer(corpus,
                          # category='democrat',
                           # category_name='positivity',
                            #not_category_name='negativity',
                             #       width_in_pixels=1000,
                              #  metadata=convention_df['speaker'])
                        # open("Convention-Visualization.html", 'wb').write(html.encode('utf-8'))
                         
                        
                         #HtmlFile = open("temp/Convention-Visualization.html", 'r', encoding='utf-8')
                         #source_code = HtmlFile.read() 
                         #print(source_code)
                         #components.html(source_code,height = 800)
                        
