import streamlit as st
import base64
from PIL import Image as PilImage
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
from langdetect import detect_langs

import scattertext as tt
import spacy
from pprint import pprint
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode

from io import BytesIO
from reportlab.lib.pagesizes import letter, landscape, A4
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Image as ReportLabImage, Spacer, BaseDocTemplate, Frame, PageTemplate
from reportlab.lib.units import inch




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
                 'About': '''## The FreeTxt/TestunRhydd tool 
         FreeTxt was developed as part of an AHRC funded collaborative
    FreeTxt supporting bilingual free-text survey  
    and questionnaire data analysis
    research project involving colleagues from
    Cardiff University and Lancaster University (Grant Number AH/W004844/1). 
    The team included PI - Dawn Knight;
    CIs - Paul Rayson, Mo El-Haj;
    RAs - Ignatius Ezeani, Nouran Khallaf and Steve Morris. 
    The Project Advisory Group included representatives from 
    National Trust Wales, Cadw, National Museum Wales,
    CBAC | WJEC and National Centre for Learning Welsh.
    -------------------------------------------------------   
    Datblygwyd TestunRhydd fel rhan o brosiect ymchwil 
    cydweithredol a gyllidwyd gan yr AHRC 
    ‘TestunRhydd: yn cefnogi dadansoddi data arolygon testun 
    rhydd a holiaduron dwyieithog’ sy’n cynnwys cydweithwyr
    o Brifysgol Caerdydd a Phrifysgol Caerhirfryn (Rhif y 
    Grant AH/W004844/1).  
    Roedd y tîm yn cynnwys PY – Dawn Knight; 
    CYwyr – Paul Rayson, Mo El-Haj; CydY 
    – Igantius Ezeani, Nouran Khallaf a Steve Morris.
    Roedd Grŵp Ymgynghorol y Prosiect yn cynnwys cynrychiolwyr 
    o Ymddiriedolaeth Genedlaethol Cymru, Amgueddfa Cymru,
    CBAC a’r Ganolfan Dysgu Cymraeg Genedlaethol.  
       '''
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
        data = open(fname, 'r', errors='ignore').read().split(r'[.\n]+') if file_source=='example' else fname.read().decode('utf8', errors='ignore').split('\n')
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

def detect_language(df):
    detected_languages = []

    # Loop through all columns in the DataFrame
    for col in df.columns:
        # Loop through all rows in the column
        for text in df[col].fillna(''):
            # Use langdetect's detect_langs to detect the language of the text
            try:
                lang_probs =  detect_langs(text)
                most_probable_lang = max(lang_probs, key=lambda x: x.prob)
                detected_languages.append(most_probable_lang.lang)
            except Exception as e:
                print(f"Error detecting language: {e}")

    # Count the number of occurrences of each language
    lang_counts = pd.Series(detected_languages).value_counts()

    # Determine the most common language in the DataFrame
    if not lang_counts.empty:
        most_common_lang = lang_counts.index[0]
    else:
        most_common_lang = None
        print("No languages detected in the DataFrame.")

    return most_common_lang



# --------------------Sentiments----------------------

###########Ployglot Welsh



def preprocess_text(text):
    # remove URLs, mentions, and hashtags
    text = re.sub(r"http\S+|@\S+|#\S+", "", text)

    # remove punctuation and convert to lowercase
    text = re.sub(f"[{re.escape(''.join(PUNCS))}]", "", text.lower())

    # remove stopwords
    text = " ".join(word for word in text.split() if word not in STOPWORDS)

    return text

# define function to analyze sentiment using Polyglot for Welsh language
@st.cache(allow_output_mutation=True)
def analyze_sentiment_welsh_polyglot(input_text):
    # preprocess input text and split into reviews
    reviews = input_text.split("\n")

    text_sentiment = []
    for review in reviews:
        review = preprocess_text(review)
        if review:
            text = Text(review, hint_language_code='cy')

            # calculate sentiment polarity per word
            sentiment_polarity_per_word = []
            for word in text.words:
                word_sentiment_polarity = word.polarity
                sentiment_polarity_per_word.append(word_sentiment_polarity)

            # calculate overall sentiment polarity
            overall_sentiment_polarity = sum(sentiment_polarity_per_word)

            # classify sentiment based on a threshold
            if overall_sentiment_polarity > 0.2:
                sentiment = "positive"
            elif overall_sentiment_polarity < -0.2:
                sentiment = "negative"
            else:
                sentiment = "neutral"

            text_sentiment.append((review, sentiment, overall_sentiment_polarity))

    return text_sentiment

from textblob import TextBlob
# define function to analyze sentiment using TextBlob for Welsh language
@st.cache(allow_output_mutation=True)
def analyze_sentiment_welsh(input_text):
    # preprocess input text and split into reviews
    reviews = input_text.split("\n")

    text_sentiment = []
    for review in reviews:
        review = preprocess_text(review)
        if review:
            # analyze sentiment using TextBlob
            text_blob = TextBlob(review)

            # calculate overall sentiment polarity
            overall_sentiment_polarity = text_blob.sentiment.polarity

            # classify sentiment based on a threshold
            if overall_sentiment_polarity > 0.2:
                sentiment = "positive"
            elif overall_sentiment_polarity < -0.2:
                sentiment = "negative"
            else:
                sentiment = "neutral"

            text_sentiment.append((review, sentiment, overall_sentiment_polarity))

    return text_sentiment


# --------------------Sentiments----------------------

###########Bert English
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
def preprocess_text(text):
    # remove URLs, mentions, and hashtags
    text = re.sub(r"http\S+|@\S+|#\S+", "", text)

    # remove punctuation and convert to lowercase
    text = re.sub(f"[{re.escape(''.join(PUNCS))}]", "", text.lower())

    # remove stopwords
    text = " ".join(word for word in text.split() if word not in STOPWORDS)

    return text



@st.cache(allow_output_mutation=True)
def analyze_sentiment(input_text,num_classes, max_seq_len=512):
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
            # Tokenize the review
            tokens = tokenizer.encode(review, add_special_tokens=True, truncation=True)

            # If the token length exceeds the maximum, split into smaller chunks
            token_chunks = []
            if len(tokens) > max_seq_len:
                token_chunks = [tokens[i:i + max_seq_len] for i in range(0, len(tokens), max_seq_len)]
            else:
                token_chunks.append(tokens)

            # Process each chunk
            sentiment_scores = []
            for token_chunk in token_chunks:
                input_ids = torch.tensor([token_chunk])
                attention_mask = torch.tensor([[1] * len(token_chunk)])

                # Run the model
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                scores = outputs.logits.softmax(dim=1).detach().numpy()[0]
                sentiment_scores.append(scores)

            # Aggregate the scores
            avg_scores = np.mean(sentiment_scores, axis=0)
            sentiment_labels = ['Very negative', 'Negative', 'Neutral', 'Positive', 'Very positive']
            sentiment_index = avg_scores.argmax()

            if num_classes == 3:
                sentiment_labels_3 = ['Negative', 'Neutral', 'Positive']
                if sentiment_index < 2:
                    sentiment_index = 0  # Negative
                elif sentiment_index > 2:
                    sentiment_index = 2  # Positive
                else:
                    sentiment_index = 1  # Neutral
                sentiment_label = sentiment_labels_3[sentiment_index]
            else:
                sentiment_label = sentiment_labels[sentiment_index]

            sentiment_score = avg_scores[sentiment_index]
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
        margin=dict(l=50, r=50, t=80, b=50),
        annotations=[
        go.layout.Annotation(
            x=0.5,
            y=1.1,
            xref='paper',
            yref='paper',
            showarrow=False,
            font=dict(size=14),
            text="The following figure displays the sentiment analysis of the data, you can press on any part of the graph to display the data",
            align='center'
        )
    ]
    )

    
    fig = go.Figure(data=data, layout=layout)
    selected_points = plotly_events(fig, select_event=True)
    
    if selected_points:
        # filter the dataframe based on the selected point
        point_number = selected_points[0]['pointNumber']
        sentiment_label = proportions.index[point_number]
        df = df[df['Sentiment Label'] == sentiment_label]
        st.write(f'The proportion of " {sentiment_label} "')
        st.dataframe(df,use_container_width = True)
    
    # update the counts and proportions based on the filtered dataframe
    counts = df['Sentiment Label'].value_counts()
    proportions = counts / counts.sum()

    # update the pie chart data
    #fig.update_traces(labels=proportions.index, values=proportions.values)

    buffer = io.StringIO()
    fig.write_html(buffer, include_plotlyjs='cdn')
    html_bytes = buffer.getvalue().encode()

    st.download_button(
        label='Download Pie Chart',
        data=html_bytes,
        file_name='Sentiment_analysis_pie.html',
        mime='text/html'
    )
    
nlp = spacy.load('en_core_web_sm-3.2.0')  
nlp.max_length = 9000000
######generate the scatter text 

def generate_scattertext_visualization(analysis):
    # Get the DataFrame with sentiment analysis results
    df = analysis
    # Parse the text using spaCy
    df['ParsedReview'] = df['Review'].apply(nlp)

    # Create a Scattertext Corpus
    corpus = tt.CorpusFromParsedDocuments(
        df,
        category_col="Sentiment Label",
        parsed_col="ParsedReview"
    ).build()
    
    
    term_scorer = tt.RankDifference()
    html = tt.produce_scattertext_explorer(
     corpus,
    category="Positive",
    not_categories=df["Sentiment Label"].unique().tolist(),
    minimum_term_frequency=5,
    pmi_threshold_coefficient=5,
    width_in_pixels=1000,
    metadata=df["Sentiment Label"],
    term_scorer=term_scorer
       ) 

    # Save the visualization as an HTML file
    with open("scattertext_visualization.html", "w") as f:
        f.write(html)

# Add a state variable to store the generated PDF data
generated_pdf_data = None


def header(canvas, doc):
    # Add logo and title in a table
    logo_path = "img/FreeTxt_logo.png" 
    logo = PilImage.open(logo_path)
    logo_width, logo_height = logo.size
    aspect_ratio = float(logo_height) / float(logo_width)
    logo = ReportLabImage(logo_path, width=100, height=int(100 * aspect_ratio))
    title_text = "Sentiemnet analysis Report"
    title_style = ParagraphStyle("Title", fontSize=20, alignment=TA_LEFT)
    title = Paragraph(title_text, title_style)
    header_data = [[logo, title]]
    header_table = Table(header_data)
    header_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (0, 0), 'LEFT'),
        ('ALIGN', (1, 0), (1, 0), 'LEFT'),
        ('VALIGN', (0, 0), (1, 0), 'TOP'),
        ('LEFTPADDING', (1, 0), (1, 0), 20),
    ]))
    w, h = header_table.wrap(doc.width, doc.topMargin)
    header_table.drawOn(canvas, doc.leftMargin, doc.height + doc.topMargin - h + 20)



    
st.markdown('''🎲 Sentiment Analyzer''')
option = st.sidebar.radio(MESSAGES[lang][0], (MESSAGES[lang][1], MESSAGES[lang][2]))
if option == MESSAGES[lang][1]: input_data = get_data()
elif option == MESSAGES[lang][2]: input_data = get_data(file_source='uploaded')
else: pass
status, data = input_data


    
if status:
        num_classes = st.radio('How do you want to categorize the sentiments?', ('3 Class Sentiments (Positive, Neutral, Negative)', '5 Class Sentiments (Very Positive, Positive, Neutral, Negative, Very Negative)'))
        num_classes = 3 if num_classes.startswith("3") else 5
        # With tabbed multiselect
        filenames = list(data.keys())
       
        for i in range(len(filenames)):
          
                _, df = data[filenames[i]]
                df = select_columns(df, key=i).astype(str)
                if df.empty:
                    st.info('''**NoColumnSelected 🤨**: Please select one or more columns to analyse.''', icon="ℹ️")
                else:
                    
                    tab1, tab2,tab3 = st.tabs(["📈 Meaning analysis",'💬 Keyword scatter','📥 Download pdf'])
                    with tab1:
                        
                        input_text = '\n'.join(['\n'.join([str(t) for t in list(df[col]) if str(t) not in STOPWORDS and str(t) not in PUNCS]) for col in df])
                        
                        language = detect_language(df)
                      
                        
                        if language == 'en':
                            sentiments = analyze_sentiment(input_text,num_classes)
                            analysis = pd.DataFrame(sentiments, columns=['Review', 'Sentiment Label', 'Sentiment Score'])
                            plot_sentiment_pie(analysis)
                            plot_sentiment(analysis)
                      
                        elif language == 'cy':
                            #sentiments = analyze_sentiment_welsh(input_text)
                            sentiments = analyze_sentiment(input_text,num_classes)
                            analysis = pd.DataFrame(sentiments, columns=['Review', 'Sentiment Label', 'Sentiment Score'])
                            plot_sentiment_pie(analysis)
                            plot_sentiment(analysis)
                       
                    with tab2:
                         #### interactive dataframe
                         gb = GridOptionsBuilder.from_dataframe(analysis)
                         gb.configure_pagination(paginationAutoPageSize=True) #Add pagination
                         gb.configure_side_bar() #Add a sidebar
                         gb.configure_selection('multiple', use_checkbox=True, groupSelectsChildren="Group checkbox select children") #Enable multi-row selection
                         gridOptions = gb.build()

                         grid_response = AgGrid(
                              analysis,
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
                        ###scattertext 
                         st.write('For better reprentation we recommend selecting 3 sentiment classes')
                         generate_scattertext_visualization(analysis)
                         HtmlFile = open("scattertext_visualization.html", 'r', encoding='utf-8')
                         source_code = HtmlFile.read() 
                         print(source_code)
                         components.html(source_code,height = 1500)
                    with tab3:
                        checkbox = st.checkbox("Generate PDF report")


                        if checkbox:

        
                        # Create the PDF
                            buffer = BytesIO()
                            doc = BaseDocTemplate(buffer, pagesize=A4,topMargin=1.5 * inch, showBoundary=0)

    # Create the frame for the content
                            frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id='normal')

    
    # Create a PageTemplate with the header
                            template = PageTemplate(id='header_template', frames=frame, onPage=header)
                            doc.addPageTemplates([template])
                            elements = []

    
       
        

    # Add a spacer between header and input text
                            elements.append(Spacer(1, 20))
        # Build PDF
	
                            doc.build(elements)
                            buffer.seek(0)
                            generated_pdf_data = buffer.read()

   # Display the download button only after generating the report
                        if generated_pdf_data:
                              st.download_button("Download PDF", generated_pdf_data, "report_positiveandnegative.pdf", "application/pdf")

                        

                        
                         
                            
                            
                         

                      
                        
