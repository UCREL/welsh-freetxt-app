import streamlit as st
import base64
from PIL import Image as PilImage
from labels import MESSAGES
from streamlit_lottie import st_lottie

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
#import en_core_web_sm
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
#########Download report

from io import BytesIO
from reportlab.lib.pagesizes import letter, landscape, A4
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Image as ReportLabImage, Spacer, BaseDocTemplate, Frame, PageTemplate
from reportlab.lib.units import inch



##Multilinguial 
import gettext
_ = gettext.gettext

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
css = '''
<style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
    font-size:2rem;
    }
</style>
'''

st.markdown(css, unsafe_allow_html=True)

@st.cache(allow_output_mutation=True)
def get_state():
    return {}
st.markdown("# Generate a summary")
st.write("---")
add_logo("img/FreeTxt_logo.png")
st.write('This tool, adapted from the Welsh Summarization project, produces a basic extractive summary of the review text from the selected columns.')

# Update with the Welsh stopwords (source: https://github.com/techiaith/ataleiriau)
en_stopwords = list(stopwords.words('english'))
cy_stopwords = open('welsh_stopwords.txt', 'r', encoding='iso-8859-1').read().split('\n') # replaced 'utf8' with 'iso-8859-1'
STOPWORDS = set(en_stopwords + cy_stopwords)
PUNCS = '''!→()-[]{};:'"\,<>./?@#$%^&*_~'''
pd.set_option('display.max_colwidth',None)

lang='en'
EXAMPLES_DIR = 'example_texts_pub'

# --- Initialising SessionState ---
if "load_state" not in st.session_state:
     st.session_state.load_state = False

# reading example and uploaded files
@st.cache(suppress_st_warning=True)
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
            fnames = st.sidebar.multiselect('Select example data file(s)', example_files,example_files[0])
            
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



summary=''
# text_rank
def text_rank_summarize(article, ratio):
  return summa_summarizer(article, ratio=ratio)

# ------------------Summarizer--------------
def run_summarizer(input_text, num,lang='en'):

    chosen_ratio_2 = st.slider(SUM_MESSAGES[f'{lang}.sb.sl'],key = f"q{num}_1", min_value=10, max_value=50, step=10)/100

    #if st.button(SUM_MESSAGES[f'{lang}.button'],key = f'bb+ {num}'):
    #if input_text and input_text!='<Rhowch eich testun (Please enter your text...)>':
    summary = text_rank_summarize(input_text, ratio=chosen_ratio_2)
    if summary:
                st.write(text_rank_summarize(input_text, ratio=chosen_ratio_2))
    else:
                st.write(sent_tokenize(text_rank_summarize(input_text, ratio=0.5))[0])
    #else:
     #       st.write("Rhowch eich testun...(Please enter your text in the above textbox)")
    return summary      
            
            
#-------------Summariser--------------
def run_summarizertxt(input_text, lang='en'):

    chosen_ratio = st.slider(SUM_MESSAGES[f'{lang}.sb.sl']+ ' ',min_value=10, max_value=50, step=10)/100

    if st.button(SUM_MESSAGES[f'{lang}.button']):
        if input_text and input_text!='<Rhowch eich testun (Please enter your text...)>' and len(input_text) > 10:
            summ = text_rank_summarize(input_text, ratio=chosen_ratio)
            if summ:
                summary = text_rank_summarize(input_text, ratio=chosen_ratio)
                st.write(summary)
            else:
                summary = sent_tokenize(text_rank_summarize(input_text, ratio=0.5))[0]
                st.write(summary)
        else:
            st.write("Rhowch eich testun...(Please enter your text in the above textbox)")
        
    
    

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
 #----------------------------------------------------------   
 # Add a state variable to store the generated PDF data
generated_pdf_data = None


def header(canvas, doc):
    # Add logo and title in a table
    logo_path = "img/FreeTxt_logo.png" 
    logo = PilImage.open(logo_path)
    logo_width, logo_height = logo.size
    aspect_ratio = float(logo_height) / float(logo_width)
    logo = ReportLabImage(logo_path, width=100, height=int(100 * aspect_ratio))
    title_text = "Sentiment Analysis Report"
    title_style = ParagraphStyle("Title", fontSize=18, alignment=TA_LEFT)
    title = Paragraph(title_text, title_style)
    header_data = [[logo, title]]
    header_table = Table(header_data)
    header_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (0, 0), 'LEFT'),
        ('ALIGN', (1, 0), (1, 0), 'LEFT'),
        ('VALIGN', (0, 0), (1, 0), 'TOP'),
        ('LEFTPADDING', (1, 0), (1, 0), 1),
    ]))
    w, h = header_table.wrap(doc.width, doc.topMargin)
    header_table.drawOn(canvas, doc.leftMargin, doc.height + doc.topMargin - h + 20)


   #--------------------------------------------------------------------------------

st.subheader('''📃 Text Summarizer''')

text = st.text_area('Rhowch eich testun (Please enter your text...)', '')
run_summarizertxt(text)



st.markdown('----')
st.subheader('''📃 File Summarizer''')
option = st.sidebar.radio(MESSAGES[lang][0], (MESSAGES[lang][1], MESSAGES[lang][2])) #, MESSAGES[lang][3]))
if option == MESSAGES[lang][1]: input_data = get_data()
elif option == MESSAGES[lang][2]: input_data = get_data(file_source='uploaded')
    # elif option == MESSAGES[lang][3]: input_data = read_example_data()
else: pass
status, data = input_data

if status:
        filenames = list(data.keys())
        #tab_titles= [f"File-{i}" for i in filenames]
        
        #tabs = st.tabs(tab_titles)
        for i in range(len(filenames)):
            
            
            
            #with tabs[i]:
                _, df = data[filenames[i]]
                df = select_columns(df, key=i).astype(str)
                if df.empty:
                    st.info('''**NoColumnSelected 🤨**: Please select one or more columns to analyse.''', icon="ℹ️")
                else:
                    tab1, tab2 = st.tabs(['📝 Summarisation','📥 Download pdf'])
                    with tab1:
                          input_text = '\n'.join(['\n'.join([str(t) for t in list(df[col]) if str(t) not in PUNCS]) for col in df])
                          summarized_text =run_summarizer(input_text[:2000],i)
                          
                           
                    with tab2:
                        download_text = st.checkbox("Include original text")
                        download_summary = st.checkbox("Include summarized text")
                        checkbox = st.checkbox("Generate PDF report")

                        if checkbox:


                           if download_text or download_summary:
                                try:
           
                                   buffer = BytesIO()
                                   doc = BaseDocTemplate(buffer, pagesize=A4, topMargin=1.5 * inch, showBoundary=0)

                                # Create the frame for the content
                                   frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id='normal')

                               # Create a PageTemplate with the header
                                       # Create a PageTemplate with the header
                                   template = PageTemplate(id='header_template', frames=frame, onPage=lambda canvas, doc: header(canvas, doc))
    
                                   doc.addPageTemplates([template])
                                   elements = []

                                 # Define the styles
                                   styles = getSampleStyleSheet()
                                   styles.add(ParagraphStyle(name='InputText', fontSize=12, textColor=colors.black))
                                   styles.add(ParagraphStyle(name='SummarizedText', fontSize=12, textColor=colors.black))

                                   if download_text:
                                 # Add the input text
                                        input_text_paragraph = Paragraph(f"Input Text:\n{input_text}", styles['InputText'])
                                        elements.append(input_text_paragraph)

                                        elements.append(Spacer(1, 20))

                                   if download_summary:
                                   # Add the summarized text
                                           
                                           summarized_text_paragraph = Paragraph(f"Summarized Text:\n{summarized_text}", styles['SummarizedText'])
                                           elements.append(summarized_text_paragraph)

            
                                   doc.build(elements)
                                   buffer.seek(0)
                                   generated_pdf_data = buffer.read()

                                    # Display the download button only after generating the report
                                   if generated_pdf_data:
                                        st.download_button("Download PDF", generated_pdf_data, "report_summarise.pdf", "application/pdf")

                                except Exception as e:
                                    st.error(f"An error occurred while generating the PDF: {str(e)}")


                        

                    
                
