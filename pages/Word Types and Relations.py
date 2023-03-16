import streamlit as st
import base64
from PIL import Image
from labels import MESSAGES

import os
import requests
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


import subprocess
import sys

import circlify ###### pip install circlify
import plotly.express as px #### pip install plotly.express
#from pyvis.network import Network
import streamlit.components.v1 as components




@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(png_file):
    with open(png_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

@st.cache(allow_output_mutation=True)
def readfile(txt_file,data):
    with open(txt_file, "w") as file:
        file.write(data)

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


	
# Remove punctuation
import string
def preprocess_punc(sent):
    x="".join([i for i in sent if i not in string.punctuation])
    ## lower case is not a feature in arabic language however i keep it to be genertic 
    return x.lower()



add_logo("img/FreeTxt_logo.png") 
st.markdown("# Word Types & Relations")
st.write("---")
st.write('''This feature uses the PyMUSAS pipeline on Spacy to generate and display POS (CyTag) tags as well as semantic (USAS) tags. 
						''')

text = "Sefydliad cyllidol yw bancwr neu fanc sy'n actio fel asiant talu ar gyfer cwsmeriaid, ac yn rhoi benthyg ac yn benthyg arian. Yn rhai gwledydd, megis yr Almaen a Siapan, mae banciau'n brif berchenogion corfforaethau diwydiannol, tra mewn gwledydd eraill, megis yr Unol Daleithiau, mae banciau'n cael eu gwahardd rhag bod yn berchen ar gwmniau sydd ddim yn rhai cyllidol. Adran Iechyd Cymru."
#text = preprocess_punc(text)
#with open ('img/data.txt', "w") as f:
 # 		f.write(text)

#text = "The Nile is a major north-flowing river in Northeastern Africa."
#data = pd.DataFrame(pd.read_csv('img/data.txt',names=[0]))
#data.to_csv('img/nn.txt')
#st.dataframe(data)
text = st.text_area("Paste text to tag", value=text)
lang_detected = detect(text)
st.write(f"Language detected: '{lang_detected}'")

   
if lang_detected == 'cy':
	###curl -F type=rest -F style=tab -F lang=cy -F text=@d.txt http://ucrel-api-01.lancaster.ac.uk/cgi-bin/pymusas.pl
	files = {
   	 'type': (None, 'rest'),
    	'style': (None, 'tab'),
    	'lang': (None, 'cy'),
    	'text': text,
		}

	response = requests.post('http://ucrel-api-01.lancaster.ac.uk/cgi-bin/pymusas.pl', files=files)
	data = response.text
	#st.text(data)
	with open('cy_tagged.txt','w') as f:
    		f.write(response.text)
	
	cy_tagged =pd.read_csv('cy_tagged.txt',sep='\t')
	cy_tagged['USAS Tags'] = cy_tagged['USAS Tags'].str.split(',').str[0]
	st.dataframe(cy_tagged,use_container_width=True)

	
	
		

        
    
elif lang_detected == 'en':
        #st.info('The English PyMUSAS tagger is still under construction...', icon='😎')
	# Load the spacy model
	nlp = spacy.load('en_core_web_sm-3.2.0')	
# Load the English PyMUSAS rule-based tagger in a separate spaCy pipeline
	english_tagger_pipeline = spacy.load('en_dual_none_contextual')
# Adds the English PyMUSAS rule-based tagger to the main spaCy pipeline
	nlp.add_pipe('pymusas_rule_based_tagger', source=english_tagger_pipeline)
	output_doc = nlp(text)
	#with st.expander('', expanded=True):
	#	st.write(f'-\t\tText\t\t\tLemma\t\t\tPOS\t\t\tUSAS Tags')
	#	for token in output_doc:
	#		st.write(f'-\t\t{token.text}\t\t\t{token.lemma_}\t\t\t{token.pos_}\t\t\t{token._.pymusas_tags}')
			
	cols = ['Text', 'Lemma', 'POS', 'USAS Tags']
	tagged_tokens = []
	for token in output_doc:
		tagged_tokens.append((token.text, token.lemma_, token.tag_, token._.pymusas_tags))
        
        # # create DataFrame using data
	tagged_tokens_df = pd.DataFrame(tagged_tokens, columns = cols)
	st.dataframe(tagged_tokens_df,use_container_width=True)
