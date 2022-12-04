import streamlit as st
import base64
from PIL import Image
from labels import MESSAGES

import os
import string
import random
import pandas as pd
import numpy as np
import streamlit as st
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

# Update with the Welsh stopwords (source: https://github.com/techiaith/ataleiriau)
en_stopwords = list(stopwords.words('english'))
cy_stopwords = open('welsh_stopwords.txt', 'r', encoding='iso-8859-1').read().split('\n') # replaced 'utf8' with 'iso-8859-1'
STOPWORDS = set(en_stopwords + cy_stopwords)
PUNCS = '''!→()-[]{};:'"\,<>./?@#$%^&*_~'''
pd.set_option('display.max_colwidth',None)

lang='en'
EXAMPLES_DIR = 'example_texts_pub'

# Load the spacy model
nlp = spacy.load('en_core_web_sm-3.2.0')
# Load the English PyMUSAS rule-based tagger in a separate spaCy pipeline
english_tagger_pipeline = spacy.load('en_dual_none_contextual')
# Adds the English PyMUSAS rule-based tagger to the main spaCy pipeline
nlp.add_pipe('pymusas_rule_based_tagger', source=english_tagger_pipeline)


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


add_logo("img/FreeTxt_logo.png") 
st.markdown("# Word Types & Relations")
st.write("---")
st.write('''This feature uses the PyMUSAS pipeline on Spacy to generate and display POS (CyTag) tags as well as semantic (USAS) tags. 
						''')

text_1 = "Sefydliad cyllidol yw bancwr neu fanc sy'n actio fel asiant talu ar gyfer cwsmeriaid, ac yn rhoi benthyg ac yn benthyg arian. Yn rhai gwledydd, megis yr Almaen a Siapan, mae banciau'n brif berchenogion corfforaethau diwydiannol, tra mewn gwledydd eraill, megis yr Unol Daleithiau, mae banciau'n cael eu gwahardd rhag bod yn berchen ar gwmniau sydd ddim yn rhai cyllidol. Adran Iechyd Cymru."

#text = "The Nile is a major north-flowing river in Northeastern Africa."

text = st.text_area("Paste text to tag", value=text_1)
lang_detected = detect(text)
st.write(f"Language detected: '{lang_detected}'")
    
if lang_detected == 'cy':
        #st.info('The Welsh PyMUSAS tagger is still under construction...', icon='😎')
	with open('img/text_test.txt', 'w+') as txt_reader:
    		string = st.text_input('ENTER TEXT', value='', max_chars=None, key=None, type='default')
    		txt_reader.write(string)
    		st.write(string)	

        #with open("img/text_test.txt", 'w', encoding='utf-8') as example_text:
            	#example_text.write(text_1)
        os.system('cat img/text_test.txt | sudo docker run -i --rm ghcr.io/ucrel/cytag:1.0.4 > welsh_text_example.tsv')
    	
        # # Load the Welsh PyMUSAS rule based tagger
        nlp = spacy.load("cy_dual_basiccorcencc2usas_contextual")

        tokens: List[str] = []
        spaces: List[bool] = []
        basic_pos_tags: List[str] = []
        lemmas: List[str] = []

        welsh_tagged_file = Path(Path.cwd(), 'welsh_text_example.tsv').resolve()

        with welsh_tagged_file.open('r', encoding='utf-8') as welsh_tagged_data:
            	for line in welsh_tagged_data:
                	line = line.strip()
                	if line:
                    		line_tags = line.split('\t')
                    		tokens.append(line_tags[1])
                    		lemmas.append(line_tags[3])
                    		basic_pos_tags.append(line_tags[4])
                    		spaces.append(True)

        # # As the tagger is a spaCy component that expects tokens, pos, and lemma
        # # we need to create a spaCy Doc object that will contain this information
        doc = Doc(Vocab(), words=tokens, tags=basic_pos_tags, lemmas=lemmas)
        output_doc = nlp(doc)

        print(f'Text\tLemma\tPOS\tUSAS Tags')
        cols = ['Text', 'Lemma', 'POS', 'USAS Tags']
        tagged_tokens = []
        for token in output_doc:
        	tagged_tokens.append((token.text, token.lemma_, token.tag_, token._.pymusas_tags))
        
        # # create DataFrame using data
        tagged_tokens_df = pd.DataFrame(tagged_tokens, columns = cols)
        tagged_tokens_df
    
#elif lang_detected == 'en':
        #st.info('The English PyMUSAS tagger is still under construction...', icon='😎')
#	output_doc = nlp(text)
#	with st.expander('', expanded=True):
#		st.write(f'-\t\tText\t\t\tLemma\t\t\tPOS\t\t\tUSAS Tags')
#		for token in output_doc:
#			st.write(f'-\t\t{token.text}\t\t\t{token.lemma_}\t\t\t{token.pos_}\t\t\t{token._.pymusas_tags}')
			
#	cols = ['Text', 'Lemma', 'POS', 'USAS Tags']
#	tagged_tokens = []
#	for token in output_doc:
#		tagged_tokens.append((token.text, token.lemma_, token.tag_, token._.pymusas_tags))
        
        # # create DataFrame using data
#	tagged_tokens_df = pd.DataFrame(tagged_tokens, columns = cols)
#	st.dataframe(tagged_tokens_df,use_container_width=True)
