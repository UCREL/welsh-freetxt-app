import streamlit as st
import base64
from PIL import Image as PilImage
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
###read the PYmusas list
pymusaslist = pd.read_csv('data/Pymusas-list.txt', names= ['USAS Tags','Equivalent Tag'])



text = st.text_area("Paste text to tag", value=text)
lang_detected = detect(text)
st.write(f"Language detected: '{lang_detected}'")

   
import requests
from requests.exceptions import ConnectionError
tagged_tokens_df= pd.DataFrame()

if lang_detected == 'cy':
    files = {
        'type': (None, 'rest'),
        'style': (None, 'tab'),
        'lang': (None, 'cy'),
        'text': (None, text),
    }

    try:
        response = requests.post('http://ucrel-api-01.lancaster.ac.uk/cgi-bin/pymusas.pl', files=files)
        data = response.text

        with open('cy_tagged.txt', 'w') as f:
            f.write(response.text)

        cy_tagged = pd.read_csv('cy_tagged.txt', sep='\t')
        cy_tagged['USAS Tags'] = cy_tagged['USAS Tags'].str.split('[,/]').str[0].str.replace('[\[\]"\']', '', regex=True)
        cy_tagged['USAS Tags'] = cy_tagged['USAS Tags'].str.split('+').str[0]
        merged_df = pd.merge(cy_tagged, pymusaslist, on='USAS Tags', how='left')
        merged_df.loc[merged_df['Equivalent Tag'].notnull(), 'USAS Tags'] = merged_df['Equivalent Tag']
        tagged_tokens_df = merged_df.drop(['Equivalent Tag'], axis=1)
        st.dataframe(tagged_tokens_df, use_container_width=True)

    except ConnectionError as e:
        st.error(f'Connection Error: {e}')

	
		

        
    
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
	
	
#########Download report

import openai
from io import BytesIO
from reportlab.lib.pagesizes import letter, landscape, A4
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Image as ReportLabImage, Spacer, BaseDocTemplate, Frame, PageTemplate
from reportlab.lib.units import inch
# Configure OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")
# Add a state variable to store the generated PDF data
generated_pdf_data = None

def generate_description(prompt):
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=300,
        n=1,
        stop=None,
        temperature=0.5,
    )
    return response.choices[0].text.strip()

def header(canvas, doc):
    # Add logo and title in a table
    logo_path = "img/FreeTxt_logo.png" 
    logo = PilImage.open(logo_path)
    logo_width, logo_height = logo.size
    aspect_ratio = float(logo_height) / float(logo_width)
    logo = ReportLabImage(logo_path, width=100, height=int(100 * aspect_ratio))
    title_text = "Semantic Tags Report"
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



# User input
input_text = text
df = tagged_tokens_df
checkbox = st.checkbox("Generate PDF report")


if checkbox:

        # Generate description for the table
        description = generate_description("Please write a paragraph to describe the following table and write a statistical summary for the column USAS tags: " + df.to_markdown())

        
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
        # Add input text
        input_text_style = ParagraphStyle("InputText", alignment=TA_LEFT)
        elements.append(Paragraph(input_text, input_text_style))
        # Add a spacer between input text and the table
        elements.append(Spacer(1, 20))
        # Add DataFrame as a table
        table_data = [df.columns.to_list()] + df.values.tolist()
        table = Table(table_data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),

            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),

            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),

            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(table)
        elements.append(Spacer(1, 20))
        # Add generated description
        description_style = ParagraphStyle("Description",  fontSize=12,alignment=TA_LEFT)
        elements.append(Paragraph(description, description_style))

        # Build PDF
	
        doc.build(elements)
        buffer.seek(0)
        generated_pdf_data = buffer.read()

   # Display the download button only after generating the report
if generated_pdf_data:
    st.download_button("Download PDF", generated_pdf_data, "report.pdf", "application/pdf")
