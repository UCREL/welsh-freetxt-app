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

import streamlit.components.v1 as components
from io import StringIO
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode
#from pandasgui import show



import streamlit as st
import base64
from PIL import Image
from labels import MESSAGES
import frequency_lists_log_likelihood as Keness

@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(png_file):
    with open(png_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

@st.cache(allow_output_mutation=True)
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
            #show(pd.DataFrame(data))
            st.dataframe(self.reviews,use_container_width=True)
            st.write('Total number of reviews: ', len(self.reviews))
            st.dataframe(self.reviews,use_container_width=True)
            st.write('Total number of reviews: ', len(self.reviews))
            HtmlFile = open("Visualization.html", 'r', encoding='utf-8')
            source_code = HtmlFile.read() 
            print(source_code)
            components.html(source_code,height = 800)

    def show_wordcloud(self, fname):
        # st.info('Word cloud ran into a technical hitch and we are fixing it...Thanks for you patience', icon='😎')
        get_wordcloud(self.reviews, fname)
    
    def show_kwic(self, fname):
        plot_kwic(self.reviews, fname)
	
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
         'About': '''## The FreeTxt tool supports bilingual (English and Welsh) free text data analysis of surveys and questionnaire responses'''
     }
 )
####

   
 
st.markdown("# Reviews analysis & illustrations")
add_logo("img/FreeTxt_logo.png") 
#st.write("---")


st.subheader('Text to Analyse')
txt = st.text_area('please past text here', '')


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

    
###to upload image
def load_image(image_file):
	img = Image.open(image_file)
	return img

def get_wordcloud (data, key):
    # st.markdown('''☁️ Word Cloud''')
    # cloud_columns = st.multiselect(
        # 'Which column do you wish to view the word cloud from?', data.columns, list(data.columns), help='Select free text columns to view the word cloud', key=f"{key}_cloud_multiselect")
    # input_data = ' '.join([' '.join([str(t) for t in list(data[col]) if t not in STOPWORDS]) for col in cloud_columns])
    # # input_data = ' '.join([' '.join([str(t) for t in list(data[col]) if t not in STOPWORDS]) for col in data])
    # for c in PUNCS: input_data = input_data.lower().replace(c,'')

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
    
   
    
    #tab2.subheader("upload mask Image")
    #image_file = tab2.file_uploader("Upload Images", type=["png","jpg","jpeg"])
    maskfile_2 = image_mask_2[tab2.selectbox('Select Cloud shape:', image_mask_2.keys(), help='Select the shape of the word cloud')]
    colors =['grey','yellow','white','black','green','blue','red']
    outlines = tab2.selectbox('Select cloud outline color ', colors, help='Select outline color word cloud')
    mask = np.array(Image.open(maskfile_2)) if maskfile_2 else maskfile_2
   
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
        
        
            
        cloud_type = tab2.selectbox('Choose Cloud category:',
            ['All words', 'Bigrams', 'Trigrams', '4-grams', 'Nouns', 'Proper nouns', 'Verbs', 'Adjectives', 'Adverbs', 'Numbers'], key= f"{key}_cloud_select")
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
        color = tab2.radio('Select image colour:', ('Color', 'Black'), key=f"{key}_cloud_radio")
        img_cols = ImageColorGenerator(mask) if color == 'Black' else None
        plt.figure(figsize=[20,15])
        plt.imshow(wordcloud.recolor(color_func=img_cols), interpolation="bilinear")
        plt.axis("off")
        with tab2:
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()
    except ValueError as err:
        with tab2:
            st.info(f'Oh oh.. Please ensure that at least one free text column is chosen: {err}', icon="🤨")
   ####generate a wordcloud based on Keness
#####English Keness
####load the Bnc Frequency list
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
    tab2.write(word_freq)
    
    freq = word_freq[['word','freq','f_Reference']].values.tolist()
    ff = [tuple(r) for r in freq]
   # ff = list(ff)
#.apply(tuple, axis=1).tolist()
    #tab2.write(ff)
	
#    ll = keness.run(ff,len(words),968267)
 #   tab2.write(ll)

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
def getTopNWords(text, topn=5, removeStops=False):
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
def plot_collocation(keyword, collocs,expander,tab):
    words, counts = zip(*collocs)
    N, total = len(counts), sum(counts)
    top_collocs_df = pd.DataFrame(collocs, columns=['word','freq'])
    plt.figure(figsize=(8,8))
    plt.xlim([-0.5, 0.5])
    plt.ylim([-0.5, 0.5])
    plt.plot([0],[0], '-o', color='blue',  markersize=25, alpha=0.7)
    plt.text(0,0, keyword, color='red', fontsize=14)
    for i in range(N):
        x, y = random.uniform((i+1)/(2*N),(i+1.5)/(2*N)), random.uniform((i+1)/(2*N), (i+1.5)/(2*N)) 
        x = x if random.choice((True, False)) else -x
        y = y if random.choice((True, False)) else -y
        plt.plot(x, y, '-og', markersize=counts[i]*10, alpha=0.3)
        plt.text(x, y, words[i], fontsize=12)
    with tab:
        with expander:
            st.dataframe(top_collocs_df,use_container_width=True)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()


########the treemap illistartion

def plot_coll(keyward, collocs,expander,tab):
    words, counts = zip(*collocs)
    
    #tab3.write(words, counts)
    
    top_collocs_df = pd.DataFrame(collocs, columns=['word','freq'])
    
    fig = px.treemap(top_collocs_df, title='Treemap chart',
                 path=[ px.Constant(keyward),'freq', 'word'], color='freq', color_continuous_scale=px.colors.sequential.GnBu, )
    fig.update_layout(margin = dict(t=50, l=25, r=25, b=25))
    with tab:
        with expander:
            
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.plotly_chart(fig,use_container_width=True)
######the network 
    n = top_collocs_df['freq'][0:30].max()
    color_dict = get_colordict('RdYlBu_r',n ,1)
    counts = list(top_collocs_df['freq'][0:30])
    top_collocs_df.insert(1, 'source', keyward)
    G= nx.from_pandas_edgelist(top_collocs_df, source = 'source', target= 'word', edge_attr='freq')
    nx.draw(G,width=top_collocs_df.freq, pos=nx.spring_layout(G, weight='draw_weight'), with_labels=True) 
    with tab:
        with expander:
            st.pyplot()
    

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
    for c in PUNCS: input_data = input_data.lower().replace(c,'')
    
    try:
        with tab3:
            topwords = [f"{w} ({c})" for w, c in getTopNWords(input_data, removeStops=True)]
            keyword = st.selectbox('Select keyword:', topwords).split('(',1)[0].strip()
            window_size = st.slider('Select window size:', 1, 10, 2)
            maxInsts = st.slider('maximum number of instances:', 5, 50, 10, 5)
        # col2_lcase = st.checkbox("Lowercase?", key='col2_checkbox')
            kwic_instances = get_kwic(input_data, keyword, window_size, maxInsts, True)
        
        #keyword_analysis = tab3.radio('Analysis:', ('Keyword in context', 'Collocation'))
        #if keyword_analysis == 'Keyword in context':
            with st.expander('Keyword in context'):
                kwic_instances_df = pd.DataFrame(kwic_instances,
                    columns =['Left context', 'Keyword', 'Right context'])
                
                kwic_instances_df.style.set_properties(column='Left context', align = 'right')
	        col1, col2, col3 = st.columns(3)
           	with col1:
                   st.dataframe(kwic_instances_df['Left context'],use_container_width=True)

                with col2:
		   st.dataframe(kwic_instances_df['Keyword'],use_container_width=True)

                with col3:
                   st.dataframe(kwic_instances_df['Right context'],use_container_width=True)

                
                st.dataframe(kwic_instances_df,use_container_width=True)       
		
            expander = st.expander('Collocation')
            with expander: #Could you replace with NLTK concordance later?
            # keyword = st.text_input('Enter a keyword:','staff')
                collocs = get_collocs(kwic_instances) #TODO: Modify to accept 'topn'               
                colloc_str = ', '.join([f"{w}[{c}]" for w, c in collocs])
                st.write(f"Collocations for '{keyword}':\n{colloc_str}")
                plot_collocation(keyword, collocs,expander,tab3)
                plot_coll(keyword, collocs,expander,tab3)
    except ValueError as err:
        with tab3:
                st.info(f'Oh oh.. Please ensure that at least one free text column is chosen: {err}', icon="🤨")


                
def plot_kwic_txt(df):
    tab6.markdown('''💬 Word location in text''')
    input_data = ' '.join([str(t) for t in df[0].split(' ') if t not in STOPWORDS])
    
    for c in PUNCS: input_data = input_data.lower().replace(c,'')
    
    try:
        with tab6:
            topwords = [f"{w} ({c})" for w, c in getTopNWords(input_data, removeStops=True)]
            keyword = st.selectbox('Select a keyword:', topwords).split('(',1)[0].strip()
            window_size = st.slider('Select the window size:', 1, 10, 2)
            maxInsts = st.slider('Maximum number of instances:', 5, 50, 10, 5)
        # col2_lcase = st.checkbox("Lowercase?", key='col2_checkbox')
            kwic_instances = get_kwic(input_data, keyword, window_size, maxInsts, True)
        
        #keyword_analysis = tab6.radio('Analysis:', ('Keyword in context', 'Collocation'))
        #if keyword_analysis == 'Keyword in context':
            with st.expander('Keyword in context'):
                kwic_instances_df = pd.DataFrame(kwic_instances,
                    columns =['Left context', 'Keyword', 'Right context'])
                kwic_instances_df.style.set_properties(column='Left context', align = 'right')
            # subset=['Left context', 'Keyword', 'Right context'],
            # kwic_instances_df
                
                st.dataframe(kwic_instances_df,use_container_width=True)
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
                st.info(f'Oh oh.. Please ensure that at least one free text column is chosen: {err}', icon="🤨")


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
             st.info('''** 🤨**: Please past text to analyse.''', icon="ℹ️")
    
        else:
       
            tab4, tab5, tab6, tab7 = st.tabs(["📈 Data View", "☁️ Keyword Cloud",'💬 Keyword in Context & Collocation', "🌳 Word Tree"])
                    ###font tabs
   
            font_css = """
                                         <style>
                                              button[data-baseweb="tab"] {
                                                 font-size: 26px;
                                                                 }
                                                          </style>
                                                      """
            st.write(font_css, unsafe_allow_html=True)           
                
        
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
    
    
        
        
        
        #tab5.subheader("upload mask Image")
        #image_file_2 = tab5.file_uploader("Upload Image", type=["png","jpg","jpeg"])
            
            maskfile = image_mask_2[tab5.selectbox('Select cloud shape:', image_mask_2.keys(), help='Select the shape of the word cloud')]
            color =['grey','yellow','white','black','green','blue','red']
            outline = tab5.selectbox('Select cloud outline color:', color, help='Select outline color word cloud')
        #if image_file_2 is not None:

			  # To See details
         #  file_details = {"filename":image_file_2.name, "filetype":image_file_2.type,"filesize":image_file_2.size}
          # img = load_image(image_file_2)
           #mask = mask = np.array(img)
        #with open(os.path.join("img",image_file_2.name),"wb") as f:
         #      f.write(image_file.getbuffer())
       
              
        #else:   
            mask = np.array(Image.open(maskfile)) if maskfile else maskfile
 
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
           feature_list = ['Data View', 'Keyword Cloud', 'Keyword in Context & Collocation', "Word Tree"]
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
                    
                    
                 
                    tab1, tab2, tab3,tab8= st.tabs(["📈 Data View", "☁️ Keyword Cloud",'💬 Keyword in Context & Collocation', " 🌳 Word Tree"])

                #if not feature_options: st.info('''**NoActionSelected☑️** Select one or more actions from the sidebar checkboxes.''', icon="ℹ️")
                    
                    analysis.show_reviews(filenames[i])
                    analysis.show_wordcloud(filenames[i])
                    analysis.show_kwic(filenames[i])
                    analysis.concordance(filenames[i])
