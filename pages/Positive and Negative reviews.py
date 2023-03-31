import streamlit as st
import base64
from PIL import Image
from labels import MESSAGES
import os
import string
import random
import pandas as pd
import numpy as np
import re
from collections import Counter
import spacy
from spacy.tokens import Doc
from spacy.vocab import Vocab
import nltk
import io
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
  

# --------------------Sentiments----------------------

###########Bert
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def preprocess_text(text):
    # remove URLs, mentions, and hashtags
    text = re.sub(r"http\S+|@\S+|#\S+", "", text)

    # remove punctuation and convert to lowercase
    text = re.sub(f"[{re.escape(''.join(PUNCS))}]", "", text.lower())

    # remove stopwords
    text = " ".join(word for word in text.split() if word not in STOPWORDS)

    return text
@st.cache(allow_output_mutation=True)
def analyze_sentiment(input_text, num_classes=3):
    # load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
    model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")

    # preprocess input text and split into reviews
    reviews = input_text.split("\n")

    # predict sentiment for each review
    sentiments = []
    for review in reviews:
        review = preprocess_text(review)
        if review:
            inputs = tokenizer.encode_plus(
                review,
                add_special_tokens=True,
                return_attention_mask=True,
                return_tensors="pt"
            )
            outputs = model(**inputs)
            scores = outputs.logits.softmax(dim=1).detach().numpy()[0]
            sentiment_labels = ['Very negative', 'Negative', 'Neutral', 'Positive', 'Very positive']
            sentiment_index = scores.argmax()
            sentiment_label = sentiment_labels[sentiment_index]
            sentiment_score = scores[sentiment_index]
            sentiments.append((review, sentiment_label, sentiment_score))

    return sentiments

#####
import plotly.graph_objs as go
import plotly.io as pio

def plot_sentiment(df):
    # count the number of reviews in each sentiment label
    counts = df['Sentiment Label'].value_counts()

    # create the bar chart
    data = [
        go.Bar(
            x=counts.index,
            y=counts.values,
            text=counts.values,
            textposition='auto',
            marker=dict(color='rgb(63, 81, 181)')
        )
    ]

    # set the layout
    layout = go.Layout(
        title='Sentiment Analysis Results',
        xaxis=dict(title='Sentiment Label'),
        yaxis=dict(title='Number of Reviews'),
        plot_bgcolor='white',
        font=dict(family='Arial, sans-serif', size=14, color='black'),
        margin=dict(l=50, r=50, t=80, b=50)
    )

    # create the figure
    fig = go.Figure(data=data, layout=layout)

    # show the plot
    st.plotly_chart(fig)
    buffer = io.StringIO()
    fig.write_html(buffer, include_plotlyjs='cdn')
    html_bytes = buffer.getvalue().encode()

    st.download_button(
            label='Download Bar Chart',
            data=html_bytes,
            file_name='Sentiment_analysis_bar.html',
            mime='text/html'
        )





from streamlit_plotly_events import plotly_events


import pandas as pd
import plotly.graph_objs as go
import streamlit as st
from streamlit_plotly_events import plotly_events

def plot_sentiment_pie(df):

    # count the number of reviews in each sentiment label
    counts = df['Sentiment Label'].value_counts()

    # calculate the proportions
    proportions = counts / counts.sum()

    # create the pie chart
    data = [
        go.Pie(
            labels=proportions.index,
            values=proportions.values,
            hole=0.4,
            marker=dict(colors=['rgb(63, 81, 181)', 'rgb(33, 150, 243)', 'rgb(255, 87, 34)'])
        )
    ]

    # set the layout
    layout = go.Layout(
        title='Sentiment Analysis Results',
        plot_bgcolor='white',
        font=dict(family='Arial, sans-serif', size=14, color='black'),
        margin=dict(l=50, r=50, t=80, b=50)
    )

    # create the figure
    fig = go.Figure(data=data, layout=layout)

    # create the event based on clicking a legend item
    event = plotly_events(fig, override_width='100%', override_height='100%')


    if event:
        # get the selected sentiment label from the event
        selected_points = event.get('data', [])
        if selected_points:
            sentiment_label = selected_points[0].get('label')
            st.write(f"Selected Sentiment Label: {sentiment_label}")
            subset_df = df[df['Sentiment Label'] == sentiment_label]
            st.write(subset_df)

    # render the plotly figure and the event details
    st.plotly_chart(fig, use_container_width=True)
    st.write(event)



   
    
st.markdown('''🎲 Sentiment Analyzer''')
option = st.sidebar.radio(MESSAGES[lang][0], (MESSAGES[lang][1], MESSAGES[lang][2]))
if option == MESSAGES[lang][1]: input_data = get_data()
elif option == MESSAGES[lang][2]: input_data = get_data(file_source='uploaded')
else: pass
status, data = input_data


    
if status:
        num_classes = st.radio('How do you want to categorize the sentiments?', ('3 Class Sentiments (Positive, Neutral, Negative)', '5 Class Sentiments (Very Positive, Positive, Neutral, Negative, Very Negative)'))
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
                        
                        input_text = '\n'.join(['\n'.join([str(t) for t in list(df[col]) if str(t) not in STOPWORDS and str(t) not in PUNCS]) for col in df])
                        sentiments = analyze_sentiment(input_text)

                    
                        analysis = pd.DataFrame(sentiments, columns=['Review', 'Sentiment Label', 'Sentiment Score'])
                        
                        plot_sentiment(analysis)
                        plot_sentiment_pie(analysis)
                        #st.dataframe(analysis, use_container_width= True)
                        
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
                        
