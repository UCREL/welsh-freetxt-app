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
import io
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from PIL import Image as PilImage
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
import math
from pathlib import Path
from typing import List
##word association
import networkx as nx
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import base64


from fpdf import FPDF


import plotly.express as px #### pip install plotly.express
from dateutil import parser
import streamlit.components.v1 as components
from io import StringIO
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode, ColumnsAutoSizeMode

from datetime import datetime
import plotly.graph_objects as go
import math
import random

import streamlit as st
import base64
from labels import MESSAGES
import frequency_lists_log_likelihood as Keness
import tempfile
#########Download report

from io import BytesIO
from reportlab.lib.pagesizes import letter, landscape, A4
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Image as ReportLabImage, Spacer, BaseDocTemplate, Frame, PageTemplate
from reportlab.lib.units import inch


@st.cache_data()
def get_base64_of_bin_file(png_file):
    with open(png_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

@st.cache_resource()
def get_state():
    return {}
### stopwords
# Update with the Welsh stopwords (source: https://github.com/techiaith/ataleiriau)
en_stopwords = list(stopwords.words('english'))
cy_stopwords = open('welsh_stopwords.txt', 'r', encoding='iso-8859-1').read().split('\n') # replaced 'utf8' with 'iso-8859-1'
STOPWORDS = set(en_stopwords + cy_stopwords)
PUNCS = '''!→()-[]{};:'"\,<>./?@#$%^&*_~'''
pd.set_option('display.max_colwidth',None)

lang='en'


# --- Initialising SessionState ---
if "load_state" not in st.session_state:
     st.session_state.load_state = False
##create the html file for the wordTree
class html:
    def __init__(self, reviews):
        self.reviews = reviews
    def create_html(self, fname,search_word):
    
    # Creating an HTML file to pass to google chart
        Func = open("GFG-1.html","w")
        sentences = ''.join(str(self.reviews.values.tolist()))
        Func.write('''<html>
  <head>
    <script type="text/javascript" src="https://www.gstatic.com/charts/loader.js"></script>
    <script type="text/javascript">
      google.charts.load('current', {packages:['wordtree']});
      google.charts.setOnLoadCallback(drawChart);

      function drawChart() {
        var data = google.visualization.arrayToDataTable(
          '''+
           sentences
             +
         ''' 
        );

        var options = {
          wordtree: {
            format: 'implicit',
            type: 'double',
            word:
            "'''          
            +
            search_word
            +
            '''"
                        
            ,
            colors: ['red', 'black', 'green']
          }
        };

        var chart = new google.visualization.WordTree(document.getElementById('wordtree_basic'));
        chart.draw(data, options);
      }
    </script>
  </head>
  <body>
    <div id="wordtree_basic" style="width: 900px; height: 500px;"></div>
  </body>
</html>

    
        ''')
        Func.close()


########### Class to Genrate PDf formate report
		
class PDF(FPDF):
    def header(self):
        # Logo
        self.image('img/FreeTxt_logo.png', 10, 8, 25)
        # font
        self.set_font('helvetica', 'B', 20)
        # Padding
        self.cell(80)
        # Title
        self.cell(30, 10, 'Collocation report', ln=1, align='C')
        # Line break
        self.ln(20)		
		
class Analysis:
    def __init__(self, reviews):
        self.reviews = reviews

    def show_reviews(self, fname):
        with tab1:
            st.markdown(f'''📄 Viewing data: `{fname}`''')
            #df = pd.DataFrame(self.reviews)
            data = self.reviews 
            #### interactive dataframe
            gb = GridOptionsBuilder.from_dataframe(data)
            gb.configure_pagination(paginationAutoPageSize=True) #Add pagination
            gb.configure_side_bar() #Add a sidebar
            gb.configure_selection('multiple', use_checkbox=True, groupSelectsChildren="Group checkbox select children") #Enable multi-row selection
            gridOptions = gb.build()

            grid_response = AgGrid(
    data,
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

		
            st.write('Total number of reviews: ', len(self.reviews))
	    
            column_list = ['date','Date','Dateandtime']
            for col in column_list:
                 if col in data.columns:
                      data['Date_sort'] = pd.to_datetime(data['Date'], format='%d/%m/%Y')
                      data= data.sort_values('Date_sort')
                      start_date = data['Date'].min()
                      end_date = data['Date'].max()
                      start_d, end_d = st.select_slider('Select a range dates', 
						      options=data['Date'].unique(),
						      value=(str(start_date), str(end_date)))
                      from dateutil import parser
                      start_d= parser.parse(start_d)
                      start_d= datetime.strftime(start_d, '%d/%m/%y')
                      end_d= parser.parse(end_d)
                      end_d = datetime.strftime(end_d, '%d/%m/%y')
                      mask = (data['Date_sort'] >= start_d) & (data['Date_sort'] <= end_d)
                      filterdf = data.loc[mask]
                      st._legacy_dataframe(filterdf)
                      st.write('filtered  number of reviews: ', len(filterdf))
           
            
            
            

    def show_wordcloud(self, fname):
        # st.info('Word cloud ran into a technical hitch and we are fixing it...Thanks for you patience', icon='😎')
        image_path=get_wordcloud(self.reviews, fname)
        return image_path
    
    def show_kwic(self, fname):
        kwic_instances_df = plot_kwic(self.reviews, fname)
        return kwic_instances_df
    def concordance(self, fname):
        with tab8:
       	    st.header('Search Word')
            search_word = st.text_input('', 'the')
            html.create_html(self, fname,search_word)
            HtmlFile = open("GFG-1.html", 'r')
            source_code = HtmlFile.read() 
            print(source_code)
            components.html(source_code,height = 800)
    
	
	
	
	
	
####add_logo
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
####


with open('style/style.css') as f:
	st.markdown(f'<style>{f.read()}</style>',unsafe_allow_html=True)

st.markdown("# Reviews analysis & illustrations")
add_logo("img/FreeTxt_logo.png") 



st.subheader('Text to analyse')
txt = st.text_area('Please paste your text here', '')


# Update with the Welsh stopwords (source: https://github.com/techiaith/ataleiriau)
en_stopwords = list(stopwords.words('english'))
cy_stopwords = open('welsh_stopwords.txt', 'r', encoding='iso-8859-1').read().split('\n') # replaced 'utf8' with 'iso-8859-1'
STOPWORDS = set(en_stopwords + cy_stopwords)
PUNCS = '''!→()-[]{};:'"\,<>./?@#$%^&*_~'''
pd.set_option('display.max_colwidth',None)

lang='en'
EXAMPLES_DIR = 'example_texts_pub'

#################################################################################
state = get_state()
#create function to get a color dictionary
def get_colordict(palette,number,start):
    pal = list(sns.color_palette(palette=palette, n_colors=number).as_hex())
    color_d = dict(enumerate(pal, start=start))
    return color_d

###ploty figure scale
def scatter(dataframe):
    df = px.data.gapminder()
    Plot_scatter = px.scatter(dataframe,y="freq", size="freq", color="word",
           hover_name="word", log_x=True, size_max=60)

    return(Plot_scatter)




# reading example and uploaded files
@st.cache_data()
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
    column_list = ['date','Date','Dateandtime']
    for col in column_list:
    	if col in data.columns:
             data['Date'] = data[col].apply(lambda x: pd.to_datetime(x).strftime('%d/%m/%Y'))


    return True, data

def get_data(file_source='example'):
    try:
        if file_source=='example':
            example_files = sorted([f for f in os.listdir(EXAMPLES_DIR) if f.startswith('Reviews')])
		# .selectbox to chang to multi selction and add the files together
            fnames = st.sidebar.multiselect('Select example data file(s)', example_files, example_files[0])
            if fnames:
			#
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



#################Get the PyMusas tags ################
###read the PYmusas list
pymusaslist = pd.read_csv('data/Pymusas-list.txt', names= ['USAS Tags','Equivalent Tag'])
def Pymsas_tags(text):
    with open('cy_tagged.txt','w') as f:
    	f.write(text)
    lang_detected = detect(text)
    if lang_detected == 'cy':
        files = {
   	 'type': (None, 'rest'),
    	'style': (None, 'tab'),
    	'lang': (None, 'cy'),
    	'text': text,
		}
        response = requests.post('http://ucrel-api-01.lancaster.ac.uk/cgi-bin/pymusas.pl', files=files)
        data = response.text
        cy_tagged =pd.read_csv('cy_tagged.txt',sep='\t')
        cy_tagged['USAS Tags'] = cy_tagged['USAS Tags'].str.split('[,/mf]').str[0].str.replace('[\[\]"\']', '', regex=True)
        cy_tagged['USAS Tags'] = cy_tagged['USAS Tags'].str.split('+').str[0]
        merged_df = pd.merge(cy_tagged, pymusaslist, on='USAS Tags', how='left')
        merged_df.loc[merged_df['Equivalent Tag'].notnull(), 'USAS Tags'] = merged_df['Equivalent Tag'] 
        merged_df = merged_df.drop(['Equivalent Tag'], axis=1)
        
    elif lang_detected == 'en':
        nlp = spacy.load('en_core_web_sm-3.2.0')	
        english_tagger_pipeline = spacy.load('en_dual_none_contextual')
        nlp.add_pipe('pymusas_rule_based_tagger', source=english_tagger_pipeline)
        output_doc = nlp(text)
        cols = ['Text', 'Lemma', 'POS', 'USAS Tags']
        tagged_tokens = []
        for token in output_doc:
             tagged_tokens.append((token.text, token.lemma_, token.tag_, token._.pymusas_tags[0]))
        tagged_tokens_df = pd.DataFrame(tagged_tokens, columns = cols)
        tagged_tokens_df['USAS Tags'] = tagged_tokens_df['USAS Tags'].str.split('[/mf]').str[0].str.replace('[\[\]"\']|-{2,}|\+{2,}', '', regex=True)
        merged_df = pd.merge(tagged_tokens_df, pymusaslist, on='USAS Tags', how='left')
        merged_df.loc[merged_df['Equivalent Tag'].notnull(), 'USAS Tags'] = merged_df['Equivalent Tag'] 
        merged_df = merged_df.drop(['Equivalent Tag'], axis=1)
        tags_to_remove = ['Unmatched', 'Grammatical bin', 'Pronouns', 'Period']
        merged_df = merged_df[~merged_df['USAS Tags'].str.contains('|'.join(tags_to_remove))]

    return(merged_df['USAS Tags'])
css = '''
<style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
    font-size:1.2rem;
    }
</style>
'''

st.markdown(css, unsafe_allow_html=True)  

    
###to upload image
def load_image(image_file):
	img = PilImage.open(image_file)
	return img

def get_wordcloud (data, key):

    tab2.markdown('''    
    ☁️ Word Cloud
    ''')
    
    layout = tab2.columns([7, 1, 4])
    cloud_columns = layout[0].multiselect(
        'Which column do you wish to view the word cloud from?', data.columns, list(data.columns), help='Select free text columns to view the word cloud', key=f"{key}_cloud_multiselect")
    input_data = ' '.join([' '.join([str(t) for t in list(data[col]) if t not in STOPWORDS]) for col in cloud_columns])
    # input_data = ' '.join([' '.join([str(t) for t in list(data[col]) if t not in STOPWORDS]) for col in data])
    for c in PUNCS: input_data = input_data.lower().replace(c,'')
    
    input_bigrams  = [' '.join(g) for g in nltk.ngrams(input_data.split(),2)]
    input_trigrams = [' '.join(g) for g in nltk.ngrams(input_data.split(),3)]
    input_4grams   = [' '.join(g) for g in nltk.ngrams(input_data.split(),4)]
    #'Welsh Flag': 'img/welsh_flag.png', 'Sherlock Holmes': 'img/holmes_silhouette.png',
    
    image_mask_2 = {'cloud':'img/cloud.png','Welsh Flag': 'img/welsh_flag.png', 'Sherlock Holmes': 'img/holmes_silhouette.png', 'national-trust':'img/national-trust-logo-black-on-white-silhouette.webp','Cadw':'img/cadw-clip.jpeg','Rectangle': None,'Tweet':'img/tweet.png','circle':'img/circle.png', 'Cadw2':'img/CadwLogo.png'}
    
   # Calculate the total number of words in the text
    Bnc_corpus=pd.read_csv('keness/Bnc.csv')
    #### Get the frequency list of the requested data using NLTK
    words = nltk.tokenize.word_tokenize(input_data)
    fdist1 = nltk.FreqDist(words)
    filtered_word_freq = dict((word, freq) for word, freq in fdist1.items() if not word.isdigit())
    column1 = list(filtered_word_freq.keys())
    column2= list(filtered_word_freq.values())
    word_freq = pd.DataFrame()
    word_freq['word']= column1
    word_freq['freq']= column2
    s = Bnc_corpus.loc[Bnc_corpus['word'].isin(column1)]
    word_freq = word_freq.merge(s, how='inner', on='word')
    #tab2.write(word_freq)
    df = word_freq[['word','freq','f_Reference']]
    
    #tab2.subheader("upload mask Image")
    #image_file = tab2.file_uploader("Upload Images", type=["png","jpg","jpeg"])
    maskfile_2 = image_mask_2[tab2.selectbox('Select Cloud shape:', image_mask_2.keys(), help='Select the shape of the word cloud')]
    colors =['grey','yellow','white','black','green','blue','red']
    outlines = tab2.selectbox('Select cloud outline color ', colors, help='Select outline color word cloud')
    mask = np.array(PilImage.open(maskfile_2)) if maskfile_2 else maskfile_2
   
    nlp = spacy.load('en_core_web_sm-3.2.0')
    doc = nlp(input_data)

    try:
        #creating wordcloud
        wc = WordCloud(
            # max_words=maxWords,
            stopwords=STOPWORDS,
            width=2000, height=1000,
		contour_color=outlines, contour_width = 1,
            relative_scaling = 0,
            mask=mask,
		
            background_color="white",
            font_path='font/Ubuntu-B.ttf'
        ).generate_from_text(input_data)
        
        
        # Allow the user to select the measure to use
	#measure = tab2.selectbox("Select a measure:", options=["Frequency","KENESS", "Log-Likelihood"])    
        cloud_type = tab2.selectbox('Choose Cloud category:',
            ['All words','Semantic Tags', 'Bigrams', 'Trigrams', '4-grams', 'Nouns', 'Proper nouns', 'Verbs', 'Adjectives', 'Adverbs', 'Numbers'], key= f"{key}_cloud_select")
        if cloud_type == 'All words':
            #wordcloud = wc.generate(input_data)
            
            # Calculate the selected measure for each word
            df = calculate_measures(df,'KENESS')
            
           
            wordcloud = wc.generate_from_frequencies(df.set_index('word')['KENESS'])

      

        elif cloud_type == 'Bigrams':
            wordcloud = wc.generate_from_frequencies(Counter(input_bigrams))        
        elif cloud_type == 'Trigrams':
            wordcloud = wc.generate_from_frequencies(Counter(input_trigrams))        
        elif cloud_type == '4-grams':
            wordcloud = wc.generate_from_frequencies(Counter(input_4grams))        
        elif cloud_type == 'Nouns':
            wordcloud = wc.generate_from_frequencies(Counter([token.text for token in doc if token.pos_ == "NOUN"]))        
        elif cloud_type == 'Proper nouns':
            wordcloud = wc.generate_from_frequencies(Counter([token.text for token in doc if token.pos_ == "PROPN"]))        
        elif cloud_type == 'Verbs':
            wordcloud = wc.generate_from_frequencies(Counter([token.text for token in doc if token.pos_ == "VERB"]))
        elif cloud_type == 'Adjectives':
            wordcloud = wc.generate_from_frequencies(Counter([token.text for token in doc if token.pos_ == "ADJ"]))
        elif cloud_type == 'Adverbs':
            wordcloud = wc.generate_from_frequencies(Counter([token.text for token in doc if token.pos_ == "ADV"]))
        elif cloud_type == 'Numbers':
            wordcloud = wc.generate_from_frequencies(Counter([token.text for token in doc if token.pos_ == "NUM"]))
        elif cloud_type == 'Semantic Tags':
            tags = Pymsas_tags(input_data)
            tags = tags.astype(str)
            wordcloud = wc.generate(' '.join(tags))

            
        else: 
            pass
        color = tab2.radio('Select image colour:', ('Color', 'Black'), key=f"{key}_cloud_radio")
        img_cols = ImageColorGenerator(mask) if color == 'Black' else None
        plt.figure(figsize=[20,15])
        wordcloud_img = wordcloud.recolor(color_func=img_cols)
        plt.imshow(wordcloud_img, interpolation="bilinear")
        plt.axis("off")

        with tab2:
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
                  wordcloud_img.to_file(tmpfile.name)
                  word_cloud_path = tmpfile.name

            img = PilImage.open(tmpfile.name)
            img_bytes = BytesIO()
            img.save(img_bytes, format='PNG')
            img_bytes = img_bytes.getvalue()
  

            # Add a download button in Streamlit to download the temporary image file
            st.download_button(
                label="Download Word Cloud Image",
                 data=img_bytes,
                 file_name="word_cloud.png",
                   mime="image/png",
                   )
    except ValueError as err:
        with tab2:
            st.info(f'Oh oh.. Please ensure that at least one free text column is chosen: {err}', icon="🤨")
    return word_cloud_path
   ####generate a wordcloud based on Keness
#####English Keness
####load the Bnc Frequency list
def calculate_measures(df,measure):

    # Convert the frequency column to an integer data type
    df['freq'] = df['freq'].astype(int)

    # Calculate the total number of words in the text
    total_words = df['freq'].sum()
    # Calculate the total number of words in the reference corpus
    ref_words = 968267
   # Calculate the KENESS and log-likelihood measures for each word
    values = []
    for index, row in df.iterrows():
        observed_freq = row['freq']
        expected_freq = row['f_Reference'] * total_words / ref_words
        if measure == 'KENESS':
            value = math.log(observed_freq / expected_freq) / math.log(2)
        elif measure == 'Log-Likelihood':
            value = 2 * (observed_freq * math.log(observed_freq / expected_freq) +
                          (total_words - observed_freq) * math.log((total_words - observed_freq) / (total_words - expected_freq)))
        values.append(value)

    # Add the measure values to the dataframe
    df[measure] = values
    return df


# ---------------Checkbox options------------------
def checkbox_container(data):
    #st.markdown('What do you want to do with the data?')
    #layout = st.columns(2)
    #if layout[0].button('Select All'):
    for i in data:
          st.session_state['dynamic_checkbox_' + i] = True
          #st.experimental_rerun()
    

def get_selected_checkboxes():
    return [i.replace('dynamic_checkbox_','') for i in st.session_state.keys() if i.startswith('dynamic_checkbox_') and 
    st.session_state[i]]

#--------------Get Top n most_common words plus counts---------------
def getTopNWords(text, topn=10, removeStops=False):
    text = text.translate(text.maketrans("", "", string.punctuation))
    text = [word for word in text.lower().split()
                if word not in STOPWORDS] if removeStops else text.lower().split()
    return Counter(text).most_common(topn) 

#---------------------keyword in context ----------------------------
def get_kwic(text, keyword, window_size=1, maxInstances=10, lower_case=False):
    text = text.translate(text.maketrans("", "", string.punctuation))
    if lower_case:
        text = text.lower()
        keyword = keyword.lower()
    kwic_insts = []
    tokens = text.split()
    keyword_indexes = [i for i in range(len(tokens)) if tokens[i].lower() == keyword.lower()]
    for index in keyword_indexes[:maxInstances]:
        left_context = ' '.join(tokens[index-window_size:index])
        target_word = tokens[index]
        right_context = ' '.join(tokens[index+1:index+window_size+1])
        kwic_insts.append((left_context, target_word, right_context))
    return kwic_insts

#---------- get collocation ------------------------
def get_collocs(kwic_insts, topn=10):
    words=[]
    for l, t, r in kwic_insts:
        words += l.split() + r.split()
    all_words = [word for word in words if word not in STOPWORDS]
    return Counter(all_words).most_common(topn)



#----------- plot collocation ------------------------
from pyvis.network import Network	
def plot_coll_14(keyword, collocs, expander, tab, output_file='network.html'):
    words, counts = zip(*collocs)
    top_collocs_df = pd.DataFrame(collocs, columns=['word', 'freq'])
    top_collocs_df.insert(1, 'source', keyword)
    top_collocs_df = top_collocs_df[top_collocs_df['word'] != keyword]  # remove row where keyword == word
    G = nx.from_pandas_edgelist(top_collocs_df, source='source', target='word', edge_attr='freq')
    n = max(counts)

    # Find the most frequent word
    most_frequent_word = max(collocs, key=lambda x: x[1])[0]

    # Create a network plot
    net = Network(notebook=True, height='750px', width='100%')

    # Adjust gravity based on frequency
    gravity = -200 * n / sum(counts)

    net.barnes_hut( gravity=gravity* 30)  # Adjust gravity to control the spread and increase repulsion force
  

    # Add nodes
    for node, count in zip(G.nodes(), counts):
        node_color = 'green' if node == most_frequent_word else 'gray' if node == keyword else 'blue'
        node_size = 100 * count / n
        font_size = max(6, int(node_size / 2))  # Adjust font size based on node size, minimum size is 6
        net.add_node(node, label=node, color=node_color, size=node_size, font={'size': font_size, 'face': 'Arial'})

    # Add edges
    for source, target, freq in top_collocs_df[['source', 'word', 'freq']].values:
        if source in net.get_nodes() and target in net.get_nodes():
            net.add_edge(source, target, value=freq)

    # Save the visualization to an HTML file
    net.save_graph(output_file)

 #-------------------------- N-gram Generator ---------------------------
def gen_ngram(text, _ngrams=2, topn=10):
    if _ngrams==1:
        return getTopNWords(text, topn)
    ngram_list=[]
    for sent in sent_tokenize(text):
        for char in sent:
            if char in PUNCS: sent = sent.replace(char, "")
        ngram_list += ngrams(word_tokenize(sent), _ngrams)
    ngram_counts = Counter(ngram_list).most_common(topn)
    sum_ngram_counts = sum([c for _, c in ngram_counts])
    return [(f"{' '.join(ng):27s}", f"{c:10d}", f"{c/sum_ngram_counts:.2f}%")
            for ng, c in ngram_counts]
#####style
th_props = [
  ('font-size', '14px'),
  ('text-align', 'left'),
  ('font-weight', 'bold'),
  ('color', '#6d6d6d'),
  ('background-color', '#eeeeef'),
  ('border','1px solid #eeeeef'),
  #('padding','12px 35px')
]

td_props = [
  ('font-size', '14px'),
  ('text-align', 'center'),
]

cell_hover_props = [  # for row hover use <tr> instead of <td>
    ('background-color', '#eeeeef')
]

headers_props = [
    ('text-align','center'),
    ('font-size','1.1em')
]
#dict(selector='th:not(.index_name)',props=headers_props)

styles = [
    dict(selector="th", props=th_props),
    dict(selector="td", props=td_props),
    dict(selector="td:hover",props=cell_hover_props),
    # dict(selector='th.col_heading',props=headers_props),
    dict(selector='th.col_heading.level0',props=headers_props),
    dict(selector='th.col_heading.level1',props=td_props)
]                
def plot_kwic(data, key):
    tab3.markdown('''💬 Word location in text''')
    
    # cloud_columns = st.multiselect(
        # 'Select your free text columns:', data.columns, list(data.columns), help='Select free text columns to view the word cloud', key=f"{key}_kwic_multiselect")
        
    # input_data = ' '.join([' '.join([str(t) for t in list(data[col]) if t not in STOPWORDS]) for col in cloud_columns])
    input_data = ' '.join([' '.join([str(t) for t in list(data[col]) if t not in STOPWORDS]) for col in data])
    nlp = spacy.load('en_core_web_sm-3.2.0')
    doc = nlp(input_data)
    for c in PUNCS: input_data = input_data.lower().replace(c,'')
    
    try:
        with tab3:
            topwords = [f"{w} ({c})" for w, c in getTopNWords(input_data, removeStops=True)]
            keyword = st.selectbox('Select keyword:', topwords).split('(',1)[0].strip()
            window_size = st.slider('Select window size:', 1, 10, 5)
            maxInsts = st.slider('Maximum number of instances:', 5, 50, 15, 5,  key="slider1_key")
        
            kwic_instances = get_kwic(input_data, keyword, window_size, maxInsts, True)
            with st.expander('Keyword_in_Context'):
                kwic_instances_df = pd.DataFrame(kwic_instances,
                    columns =['Left context', 'Keyword', 'Right context'])
                #kwic_instances_df.style.hide_index()
                
          
		   #### interactive dataframe
                gb = GridOptionsBuilder.from_dataframe(kwic_instances_df)
              
                gb.configure_column("Left context", cellClass ='text-right', headerClass= 'ag-header-cell-text' )
		
                gb.configure_column("Keyword", cellClass ='text-center', cellStyle= {
                   'color': 'red', 
                   'font-weight': 'bold'  })
                gb.configure_column("Right context", cellClass ='text-left')
                gb.configure_pagination(paginationAutoPageSize=True) #Add pagination
                gb.configure_side_bar() #Add a sidebar
                gb.configure_selection('multiple', use_checkbox=True, groupSelectsChildren="Group checkbox select children") #Enable multi-row selection
                gridOptions = gb.build()

                grid_response = AgGrid(
                kwic_instances_df,
                gridOptions=gridOptions,
                   data_return_mode='AS_INPUT', 
                   update_mode='MODEL_CHANGED', 
                   fit_columns_on_grid_load=False,
    
                   enable_enterprise_modules=True,
		   key='select_grid',
                   height=350, width= '100%',
                   columns_auto_size_mode=ColumnsAutoSizeMode.FIT_CONTENTS,
                     reload_data=True
                      )
                data = grid_response['data']
                selected = grid_response['selected_rows'] 
                df = pd.DataFrame(selected) 
                #st.write(df)
		
            expander = st.expander('Collocation')
            with expander: #Could you replace with NLTK concordance later?
            # keyword = st.text_input('Enter a keyword:','staff')
                Word_type = st.selectbox('Choose word type:',
                 ['All words', 'Nouns', 'Proper nouns', 'Verbs', 'Adjectives', 'Adverbs', 'Numbers'], key= f"{key}_type_select")
                collocs = get_collocs(kwic_instances)
                colloc_str = ', '.join([f"{w} [{c}]" for w, c in collocs])
                words = nlp(colloc_str)
                if Word_type == 'All words':
                       st.write(f"Collocations for '{keyword}':\n{colloc_str}")
                    
                elif Word_type == 'Nouns':
                       
                       collocs = [token.text for token in words if token.pos_ == "NOUN"]
                       st.write(collocs)
                       st.write(f"Collocations for '{keyword}':\n{colloc_str}")
                elif Word_type == 'Proper nouns':
                       collocs = [token.text for token in words if token.pos_ == "PROPN"]
                       st.write(collocs)
                       st.write(f"Collocations for '{keyword}':\n{colloc_str}")
                
                elif Word_type == 'Verbs':
                       collocs = [token.text for token in words if token.pos_ == "VERB"]
                       st.write(collocs)
                       st.write(f"Collocations for '{keyword}':\n{colloc_str}")
                elif Word_type == 'Adjectives':
                       collocs = [token.text for token in words if token.pos_ == "ADJ"]
                       st.write(collocs)
                       st.write(f"Collocations for '{keyword}':\n{colloc_str}")
                elif Word_type == 'Adverbs':
                       collocs = [token.text for token in words if token.pos_ == "ADV"]
                       st.write(collocs)
                       st.write(f"Collocations for '{keyword}':\n{colloc_str}")
                elif Word_type == 'Numbers':
                       collocs = [token.text for token in words if token.pos_ == "NUM"]
                       st.write(collocs)
                       st.write(f"Collocations for '{keyword}':\n{colloc_str}")
                else: 
                      pass
		
             
                plot_coll_14(keyword, collocs, expander, tab3,output_file='network_output.html')
                with open('network_output.html', 'r', encoding='utf-8') as f:
                         html_string = f.read()
                components.html(html_string, width=800, height=750, scrolling=True)
                with open('network_output.html', 'rb') as f:
                        html_bytes = f.read()
                st.download_button(
                  label='Download Collocation Graph',
                  data=html_bytes,
                  file_name='network_output.html',
                   mime='text/html'
                       )
    
     
                
    except ValueError as err:
        with tab3:
                st.info(f'Oh oh.. Please ensure that at least one free text column is chosen: {err}', icon="🤨")
    return kwic_instances_df
 #----------------------Generate Pdf--------------------------   
 # Add a state variable to store the generated PDF data
generated_pdf_data = None


def header(canvas, doc):
    # Add logo and title in a table
    logo_path = "img/FreeTxt_logo.png" 
    logo = PilImage.open(logo_path)
    logo_width, logo_height = logo.size
    aspect_ratio = float(logo_height) / float(logo_width)
    logo = ReportLabImage(logo_path, width=100, height=int(100 * aspect_ratio))
    title_text = "Text Analysis Report"
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
   #--------------------------------------------------------------------------------
                
def plot_kwic_txt(df):
    tab6.markdown('''💬 Word location in text''')
    input_data = ' '.join([str(t) for t in df[0].split(' ') if t not in STOPWORDS])
    
    for c in PUNCS: input_data = input_data.lower().replace(c,'')
    
    try:
        with tab6:
            topwords = [f"{w} ({c})" for w, c in getTopNWords(input_data, removeStops=True)]
            keyword = st.selectbox('Select a keyword:', topwords).split('(',1)[0].strip()
            window_size = st.slider('Select the window size:', 1, 10, 5)
            maxInsts = st.slider('Maximum number of instances :', 5, 50, 15, 5, key="slider2_key")
        # col2_lcase = st.checkbox("Lowercase?", key='col2_checkbox')
            kwic_instances = get_kwic(input_data, keyword, window_size, maxInsts, True)
        
        #keyword_analysis = tab6.radio('Analysis:', ('Keyword in context', 'Collocation'))
        #if keyword_analysis == 'Keyword in context':
            with st.expander('Keyword in Context'):
                kwic_instances_df = pd.DataFrame(kwic_instances,
                columns =['Left context', 'Keyword', 'Right context'])
                #kwic_instances_df.style.hide_index()
                
          
		   #### interactive dataframe
                gb = GridOptionsBuilder.from_dataframe(kwic_instances_df)
              
                gb.configure_column("Left context", cellClass ='text-right', headerClass= 'ag-header-cell-text' )
		
                gb.configure_column("Keyword", cellClass ='text-center', cellStyle= {
                   'color': 'red', 
                   'font-weight': 'bold'  })
                gb.configure_column("Right context", cellClass ='text-left')
                gb.configure_pagination(paginationAutoPageSize=True) #Add pagination
                gb.configure_side_bar() #Add a sidebar
                gb.configure_selection('multiple', use_checkbox=True, groupSelectsChildren="Group checkbox select children") #Enable multi-row selection
                gridOptions = gb.build()

                grid_response = AgGrid(
                kwic_instances_df,
                gridOptions=gridOptions,
                   data_return_mode='AS_INPUT', 
                   update_mode='MODEL_CHANGED', 
                   fit_columns_on_grid_load=False,
    
                   enable_enterprise_modules=True,
		   key='select_grid_2',
                   height=350, width= '100%',
                   columns_auto_size_mode=ColumnsAutoSizeMode.FIT_CONTENTS,
                     reload_data=True
                      )
                data = grid_response['data']
                selected = grid_response['selected_rows'] 
                df = pd.DataFrame(selected)
            expander = st.expander('collocation')
            with expander: #Could you replace with NLTK concordance later?
            # keyword = st.text_input('Enter a keyword:','staff')
                collocs = get_collocs(kwic_instances) #TODO: Modify to accept 'topn'               
                colloc_str = ', '.join([f"{w}[{c}]" for w, c in collocs])
                st.write(f"Collocations for '{keyword}':\n{colloc_str}")
                plot_collocation(keyword, collocs,expander,tab6)
                plot_coll(keyword, collocs,expander,tab6)
    except ValueError as err:
        with tab6:
                st.info(f'Please ensure that at least one free text column is chosen: {err}', icon="🤨")
    return kwic_instances_df

if st.button('Analysis') or st.session_state.load_state:
    st.session_state.load_state = True
    #st.write(txt)
    area =[]
    if len(txt) < 10:
        st.write("Rhowch eich testun...(Please enter your text in the above textbox)")
    else:
        area.append(txt)
   
            
        df = pd.DataFrame(area)
        df.columns =['Review']
    
       
        df = df['Review'].dropna(how='all').drop_duplicates()
   
        if df.empty:
             st.info('''** 🤨**: Please paste text to analyse.''', icon="ℹ️")
    
        else:
       
            tab4, tab5, tab6, tab7, tab10 = st.tabs(["📈 Data View", "☁️ Keyword Cloud",'💬 Keyword in Context & Collocation', "🌳 Word Tree", '📥 Generate pdf report'])
          
                
        
        ##show review
            tab4.dataframe(df ,use_container_width=True)
        ###show word cloud
        
            tab5.markdown('''    
             ☁️ Word Cloud
            ''')
    
            layout = tab5.columns([7, 1, 4])
            input_data = ' '.join([str(t) for t in df[0].split(' ') if t not in STOPWORDS])
        
            for c in PUNCS: input_data = input_data.lower().replace(c,'')
    
            input_bigrams  = [' '.join(g) for g in nltk.ngrams(input_data.split(),2)]
            input_trigrams = [' '.join(g) for g in nltk.ngrams(input_data.split(),3)]
            input_4grams   = [' '.join(g) for g in nltk.ngrams(input_data.split(),4)]
    
            image_mask_2 = {'Welsh Flag': 'img/welsh_flag.png', 'Sherlock Holmes': 'img/holmes_silhouette.png', 'national-trust':'img/national-trust-logo-black-on-white-silhouette.webp','Cadw':'img/cadw-clip.jpeg','Rectangle': None,'cloud':'img/cloud.png','Circle':'img/circle.png','Tweet':'img/tweet.png','Cadw2':'img/CadwLogo.png'}
    
    
        
            
            maskfile = image_mask_2[tab5.selectbox('Select cloud shape:', image_mask_2.keys(), help='Select the shape of the word cloud')]
            color =['grey','yellow','white','black','green','blue','red']
            outline = tab5.selectbox('Select cloud outline color:', color, help='Select outline color word cloud')
    
       
              
       
            mask = np.array(PilImage.open(maskfile)) if maskfile else maskfile
 
            nlp = spacy.load('en_core_web_sm-3.2.0')
            doc = nlp(input_data)

            try:
            #creating wordcloud
               wc = WordCloud(
            # max_words=maxWords,
                stopwords=STOPWORDS,
                width=2000, height=1000,
                relative_scaling = 0,
		contour_color=outline, contour_width =1,
                mask=mask,
                background_color="white",
                font_path='font/Ubuntu-B.ttf'
                ).generate(input_data)
        
        #, key= f"{key}_cloud_select"
            
               cloud_type = tab5.selectbox('Choose cloud category:',
                            ['All words', 'Bigrams', 'Trigrams', '4-grams', 'Nouns', 'Proper nouns', 'Verbs', 'Adjectives', 'Adverbs', 'Numbers'])
               if cloud_type == 'All words':
                  wordcloud = wc.generate(input_data)        
               elif cloud_type == 'Bigrams':
                  wordcloud = wc.generate_from_frequencies(Counter(input_bigrams))        
               elif cloud_type == 'Trigrams':
                  wordcloud = wc.generate_from_frequencies(Counter(input_trigrams))        
               elif cloud_type == '4-grams':
                  wordcloud = wc.generate_from_frequencies(Counter(input_4grams))        
               elif cloud_type == 'Nouns':
                  wordcloud = wc.generate_from_frequencies(Counter([token.text for token in doc if token.pos_ == "NOUN"]))        
               elif cloud_type == 'Proper nouns':
                  wordcloud = wc.generate_from_frequencies(Counter([token.text for token in doc if token.pos_ == "PROPN"]))        
               elif cloud_type == 'Verbs':
                  wordcloud = wc.generate_from_frequencies(Counter([token.text for token in doc if token.pos_ == "VERB"]))
               elif cloud_type == 'Adjectives':
                  wordcloud = wc.generate_from_frequencies(Counter([token.text for token in doc if token.pos_ == "ADJ"]))
               elif cloud_type == 'Adverbs':
                  wordcloud = wc.generate_from_frequencies(Counter([token.text for token in doc if token.pos_ == "ADV"]))
               elif cloud_type == 'Numbers':
                  wordcloud = wc.generate_from_frequencies(Counter([token.text for token in doc if token.pos_ == "NUM"]))
               else: 
                  pass
            #, key=f"{key}_cloud_radio"
            
               color = tab5.radio('switch image colour:', ('Color', 'Black'))
               img_cols = ImageColorGenerator(mask) if color == 'Black' else None
               plt.figure(figsize=[20,15])
            
               plt.imshow(wordcloud.recolor(color_func=img_cols), interpolation="bilinear")
               plt.axis("off")
               with tab5:
                  st.set_option('deprecation.showPyplotGlobalUse', False)
                  st.pyplot()
            except ValueError as err:
               with tab5:
                  st.info(f'Oh oh.. Please ensure that at least one free text column is chosen: {err}', icon="🤨")
        
            with tab6:
                plot_kwic_txt(df)
            with tab10:
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
                              st.download_button("Download PDF", generated_pdf_data, "report_TextAnalysis.pdf", "application/pdf")

    
	

st.markdown("""---""")
st.subheader('File to Analyse')
#st.markdown('''🔍 Free Text Visualizer''')
option = st.sidebar.radio(MESSAGES[lang][0], (MESSAGES[lang][1], MESSAGES[lang][2])) #, MESSAGES[lang][3]))
if option == MESSAGES[lang][1]: input_data = get_data()
elif option == MESSAGES[lang][2]: input_data = get_data(file_source='uploaded')
    # elif option == MESSAGES[lang][3]: input_data = read_example_data()
else: pass

status, data = input_data
if status:
    if 'feature_list' not in st.session_state.keys():
           feature_list = ['Data View', 'Keyword Cloud', 'Keyword in Context & Collocation', "Word Tree",'Generate pdf report']
           st.session_state['feature_list'] = feature_list
    else:
     feature_list = st.session_state['feature_list']
     checkbox_container(feature_list)
     feature_options = get_selected_checkboxes()
    
 
  

    
    
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
                
                    analysis = Analysis(df)
                    
                    
                 
                    tab1, tab2, tab3,tab8, tab9= st.tabs(["📈 Data View", "☁️ Keyword Cloud",'💬 Keyword in Context & Collocation', " 🌳 Word Tree", '📥 Generate pdf report'])

                #if not feature_options: st.info('''**NoActionSelected☑️** Select one or more actions from the sidebar checkboxes.''', icon="ℹ️")
                    
                    analysis.show_reviews(filenames[i])
                    word_cloud_path = analysis.show_wordcloud(filenames[i])
                    Keyword_context = analysis.show_kwic(filenames[i])
                    analysis.concordance(filenames[i])
                    with tab9:
			   # Check if the DataFrame exists
                         if analysis is not None:
        #####pdf_generator
                             full_data_table_checkbox = st.checkbox("Include Full Data Table")
                             word_cloud_checkbox = st.checkbox("Include Word Cloud Image")
                             keyword_context_table_checkbox = st.checkbox("Include Keyword in Context Table")
                             #keyword_context_diagram_checkbox = st.checkbox("Include Keyword in Context Diagram")
                             generate_pdf_checkbox = st.checkbox("Generate PDF report")
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
                            
                        
                         if full_data_table_checkbox:
          
                                    

                            column_names = ['Data']
                            table_data = [column_names] + df.values.tolist()
                            col_widths = [200, 100, 100]  # Adjust these values according to your needs
                            wrapped_cells = []

                            styles = getSampleStyleSheet()
                            cell_style_normal = ParagraphStyle(name='cell_style_normal', parent=styles['Normal'], alignment=1)
                            cell_style_header = ParagraphStyle(name='cell_style_header', parent=styles['Normal'], alignment=1, textColor=colors.whitesmoke, backColor=colors.grey, fontName='Helvetica-Bold', fontSize=14, spaceAfter=12)

                            wrapped_data = []
                            for row in table_data:
                                  wrapped_cells = []
                                  for i, cell in enumerate(row):
                                       cell_style = cell_style_header if len(wrapped_data) == 0 else cell_style_normal
                                       wrapped_cell = Paragraph(str(cell), style=cell_style)
                                       wrapped_cells.append(wrapped_cell)
                                  wrapped_data.append(wrapped_cells)
                            table = Table(wrapped_data, colWidths=col_widths)

                            table.setStyle(TableStyle([
                                  ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                              ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),

                             ('ALIGN', (0, 0), (-1, -1), 'CENTER'),

                                  ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                                  ('FONTSIZE', (0, 0), (-1, 0), 14),

                                   ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                                           ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                                  ('GRID', (0, 0), (-1, -1), 1, colors.black),
       
                                 ('VALIGN', (0, 0), (-1, -1), 'TOP'),

                                         ]))
                            elements.append(table)
                            elements.append(Spacer(1, 20))

                         if word_cloud_checkbox:
         
	
                        
                      # Load the image with PIL for ReportLab
                            img = PilImage.open(word_cloud_path)
                                # Load the image with PIL for ReportLab

                          # Convert the PIL Image object to binary data (bytes)
                            img_bytes = BytesIO()
                            img.save(img_bytes, format='PNG')
                            img_bytes.seek(0)

                               # Add the Word Cloud image to the PDF
                            word_cloud_image = ReportLabImage(img_bytes, width=325, height=250)
                            elements.append(word_cloud_image)
                            elements.append(Spacer(1, 20))


                         if keyword_context_table_checkbox:
           
                            columns = ['Left context', 'Keyword', 'Right context']
                            table_data = [columns] + Keyword_context.values.tolist()
                            col_widths = [150, 100, 150] 
                            wrapped_cells = []

                            styles = getSampleStyleSheet()
                            cell_style_normal = ParagraphStyle(name='cell_style_normal', parent=styles['Normal'], alignment=1)
                            cell_style_header = ParagraphStyle(name='cell_style_header', parent=styles['Normal'], alignment=1, textColor=colors.whitesmoke, backColor=colors.grey, fontName='Helvetica-Bold', fontSize=14, spaceAfter=12)

                            wrapped_data = []
                            for row in table_data:
                                     wrapped_cells = []
                                     for i, cell in enumerate(row):
                                           cell_style = cell_style_header if len(wrapped_data) == 0 else cell_style_normal
                                           wrapped_cell = Paragraph(str(cell), style=cell_style)
                                           wrapped_cells.append(wrapped_cell)
                                     wrapped_data.append(wrapped_cells)
                            table = Table(wrapped_data, colWidths=col_widths)

                            table.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    ('TEXTCOLOR', (1, 1), (1, -1), colors.red),

    ('ALIGN', (0, 1), (0, -1), 'RIGHT'),
    ('ALIGN', (1, 0), (1, -1), 'CENTER'),
    ('ALIGN', (2, 1), (2, -1), 'LEFT'),

    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, 0), 14),

    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
    ('GRID', (0, 0), (-1, -1), 1, colors.black),

    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                                            ]))

                            elements.append(table)
                            elements.append(Spacer(1, 20))



                         #if keyword_context_diagram_checkbox:
          
                          #        pass

      
                         if generate_pdf_checkbox:

        
                        
     			   # Build PDF
	
                            doc.build(elements)
                            buffer.seek(0)
                            generated_pdf_data = buffer.read()

  			 # Display the download button only after generating the report
                         if generated_pdf_data:
                              st.download_button("Download PDF", generated_pdf_data, "report_TextAnalysis.pdf", "application/pdf")

    