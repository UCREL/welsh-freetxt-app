import os
import string
import spacy
import nltk
import random
import en_core_web_sm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import networkx as nx
from PIL import Image
from io import StringIO
from nltk import word_tokenize, sent_tokenize, ngrams
from collections import Counter
from summa.summarizer import summarize as summa_summarizer
from wordcloud import WordCloud, ImageColorGenerator
from nltk.corpus import stopwords
nltk.download('punkt') # one time execution
nltk.download('stopwords')

random.seed(10)

# Update with the Welsh stopwords (source: https://github.com/techiaith/ataleiriau)
en_stopwords = list(stopwords.words('english'))
cy_stopwords = open('welsh_stopwords.txt', 'r', encoding='utf8').read().split('\n')
STOPWORDS = set(en_stopwords + cy_stopwords)
PUNCS = '''!→()-[]{};:'"\,<>./?@#$%^&*_~'''

EXAMPLES_DIR = 'example_texts_pub'
# EXAMPLES_DIR = 'example_texts_cadw'

## Define summarizer models
# text_rank
def text_rank_summarize(article, ratio):
  return summa_summarizer(article, ratio=ratio)

#helper functions---------------------------------------------------

#------------------------- uploading file ---------------------------
def uploadfile():
    uploaded_file = st.file_uploader("Choose a text file")
    if uploaded_file is not None:
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        return stringio.read()
    else:
        return '<Please upload your file ...>'

def upload_multiple_files():
    uploaded_files = st.file_uploader(
                    "Select file(s) to upload", accept_multiple_files=True)
    bytes_data = ''
    for uploaded_file in uploaded_files:
        bytes_data += uploaded_file.read().decode("utf-8") 
    return bytes_data
         # st.write("filename:", uploaded_file.name)
         # st.write(bytes_data)

#--------------Get Top n most_common words plus counts---------------
@st.cache
def getTopNWords(t, n=5):
    t = [w for w in t.lower().split() if (w not in STOPWORDS and w not in PUNCS)]
    return Counter(t).most_common(n)
    # return [f"{w} ({c})" for w, c in Counter(t).most_common(n)]

#------------------------ keyword in context ------------------------
@st.cache
def get_kwic(text, keyword, window_size=1, maxInstances=10, lower_case=False):
    text = text.translate(text.maketrans("", "", string.punctuation))
    if lower_case:
        text = text.lower()
        keyword = keyword.lower()
    kwic_insts = []
    tokens = text.split()
    keyword_indexes = [i for i in range(len(tokens)) if tokens[i].lower() ==keyword.lower()]
    for index in keyword_indexes[:maxInstances]:
        left_context = ' '.join(tokens[index-window_size:index])
        target_word = tokens[index]
        right_context = ' '.join(tokens[index+1:index+window_size+1])
        kwic_insts.append((left_context, target_word, right_context))
    return kwic_insts

#------------------------ get collocation ------------------------
@st.cache
def get_collocs(kwic_insts, topn=10):
    words=[]
    for l, t, r in kwic_insts:
        words += l.split() + r.split()
    all_words = [word for word in words if word not in STOPWORDS]
    return Counter(all_words).most_common(topn)

#------------------------ plot collocation ------------------------
@st.cache(suppress_st_warning=True)
def plot_collocation(keyword, collocs):
    words, counts = zip(*collocs)
    N, total = len(counts), sum(counts)
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
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

#-------------------------- N-gram Generator ---------------------------
@st.cache
def gen_ngram(text, n=2, top=10):
    _ngrams=[]
    if n==1:
        return getTopNWords(text, top)
    for sent in sent_tokenize(text):
        for char in sent:
            if char in PUNCS: sent = sent.replace(char, "")
        _ngrams += ngrams(word_tokenize(sent),n)
    return [(f"{' '.join(ng):>27s}", c) 
            for ng, c in Counter(_ngrams).most_common(top)]

#apps------------------------------------------------------------------
def run_summarizer():
    language = st.sidebar.selectbox('Newid iaith (Change language):', ['Cymraeg', 'English'])
    with st.expander("ℹ️ - About this app", expanded=False):
        st.markdown(
            """     
            -   This tool adapts the app from the [Welsh Summarization] (https://github.com/UCREL/welsh-summarization-dataset) project!
            -   It performs simple extractive summarisation with the [TextRank]() alrogithm.
            """
        )

    if language=='Cymraeg':
        st.markdown('### 🌷 Adnodd Creu Crynodebau')
        st.markdown("#### Rhowch eich testun isod:")
        option = st.sidebar.radio('Sut ydych chi am fewnbynnu eich testun?', ('Defnyddiwch destun enghreifftiol', 'Rhowch eich testun eich hun', 'Llwythwch ffeil testun i fyny'))
        if option == 'Defnyddiwch destun enghreifftiol':
           example_fname = st.sidebar.selectbox('Select example text:', sorted([f for f in os.listdir(EXAMPLES_DIR)
                                                  if f.startswith(('cy','ex'))]))

           with open(os.path.join(EXAMPLES_DIR, example_fname), 'r', encoding='utf8') as example_file:
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
        st.markdown('### 🌷 Welsh Summary Creator')
        st.markdown("#### Enter your text below:")
        option = st.sidebar.radio('How do you want to input your text?', ('Use an example text', 'Paste a copied', 'Upload a text file'))
        if option == 'Use an example text':           
           example_fname = st.sidebar.selectbox('Select example text:', sorted([f for f in os.listdir(EXAMPLES_DIR)
                                                  if f.startswith(('en','ex'))]))
           with open(os.path.join(EXAMPLES_DIR, example_fname), 'r', encoding='utf8') as example_file:
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

def run_visualizer():
    # language = st.sidebar.selectbox('Newid iaith (Change language):', ['Cymraeg', 'English'])
    with st.expander("ℹ️ - About Visualizer", expanded=False):
        st.markdown(
            """
            The `Visualizer` tool provides: 
            * N-gram Frequency: **input**: `text`, `ngrams`, `top ngrams:(default=10)`; **Output**: *list of tuples:* (`NGram`, `Counts`)
            * Keyword in Context (KWIC): **input**: `text`, `keyword`, `window_size:(default=1)`, `maxInstances=(default=10)`, `lower_case=(False)`; **Output**: *list of tuples:* (`left_context`, `keyword`, `right_context`)
            * Word Cloud: **input**: `text`, `num_words`, `color`; **Output**: Word Cloud image
            """
        )

    # st.markdown('### 🔍 Visualization')
    option = st.sidebar.radio('How do you want to input your text?', ('Use an example text', 'Paste copied text', 'Upload files'))
    if option == 'Use an example text':
       example_fname = st.sidebar.selectbox('Select example text:', sorted([f for f in os.listdir(EXAMPLES_DIR) if f.startswith(('ex'))]))
       
       with open(os.path.join(EXAMPLES_DIR, example_fname), 'r', encoding='utf8') as example_file:
           example_text = example_file.read()
           input_text = st.text_area('Visualize example text in the box:', example_text, height=150)
    elif option == 'Upload files':
        text = upload_multiple_files()
        input_text = st.text_area('Visualize uploaded text:', text, height=150)
    else:
        input_text = st.text_area('Type or paste your text into the text box:', '<Please enter your text...>', height=150)

    img_cols = None
    col0, col1, col2 = st.columns(3)
    
    col0.markdown("**NGram Frequency**")
    with col0:
        # keyword = st.text_input('Enter a keyword:')
        ngrms = st.slider('Select ngrams:', 1, 5, 1)
        topn = st.slider('Top ngrams:', 10, 50, 10)
        # col0_lcase = st.checkbox("Lowercase?")
        # if col0_lcase: input_text = input_text.lower()

        top_ngrams = gen_ngram(input_text, ngrms, topn)
        top_ngrams_df = pd.DataFrame(top_ngrams,
            columns =['NGrams', 'Counts'])
        st.dataframe(top_ngrams_df)
    
    with col1:
        st.markdown("**Word Cloud**")
        mask = np.array(Image.open('img/welsh_flag.png'))      
        maxWords = st.slider('Maximum number of words:', 10, 300, 300, 10)
        
        if input_text:
            nlp = spacy.load('en_core_web_sm')
            doc = nlp(input_text)
            nouns = Counter([token.lemma_ for token in doc if token.pos_ == "NOUN"])
            verbs = Counter([token.lemma_ for token in doc if token.pos_ == "VERB"])
        
            #creating wordcloud
            wc = WordCloud(
                max_words=maxWords,
                stopwords=STOPWORDS,
                width=2000, height=1000,
                # contour_color= "black", 
                relative_scaling = 0,
                mask=mask,
                background_color="white",
                font_path='font/Ubuntu-B.ttf'
            )#.generate(input_text)
            
            cloud_type = col1.selectbox('Choose cloud type:', ['All words', 'Nouns', 'Verbs'])
            if cloud_type == 'All words':
                wordcloud = wc.generate(input_text)        
            elif cloud_type == 'Nouns':
                wordcloud = wc.generate_from_frequencies(nouns)        
            else: 
                wordcloud = wc.generate_from_frequencies(verbs)

            color = st.radio('Switch image colour:', ('Color', 'Black'))
            img_cols = ImageColorGenerator(mask) if color == 'Black' else None
            
            # image_colors = ImageColorGenerator(mask)
            plt.figure(figsize=[20,15])
            
            # plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear")
            plt.imshow(wordcloud.recolor(color_func=img_cols), interpolation="bilinear")
            plt.axis("off")
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()
    
    col2.markdown("**Keyword in Context**")
    with col2: #Could you replace with NLTK concordance later?
        if input_text:
            keyword_analysis = st.radio('Keyword Anaysis:', ('Keyword in context', 'Collocation'))
            topwords = [f"{w} ({c})" for w, c in getTopNWords(input_text)]
            keyword = st.selectbox('Select a keyword:', topwords).split('(',1)[0].strip()
            window_size = st.slider('Select the window size:', 1, 10, 2)
            maxInsts = st.slider('Maximum number of instances:', 5, 50, 10, 5)
            col2_lcase = st.checkbox("Lowercase?")
            kwic_instances = get_kwic(input_text, keyword, window_size, maxInsts, col2_lcase)
            if keyword_analysis == 'Keyword in context':
                kwic_instances_df = pd.DataFrame(kwic_instances,
                    columns =['left context', 'keyword', 'right context'])
                col2.dataframe(kwic_instances_df)
            else: #Could you replace with NLTK concordance later? 
                # keyword = st.text_input('Enter a keyword:','staff')
                collocs = get_collocs(kwic_instances) #TODO: Modify to accept 'topn'
                colloc_str = ', '.join([f"{w}[{c}]" for w, c in collocs])
                col2.write(f"Collocations for '{keyword}':\n{colloc_str}")
                plot_collocation(keyword, collocs)

def run_analyze():
    with st.expander("ℹ️ - About Analyzer", expanded=False):
        st.markdown(
            """
            This tool is still at the developmental stage. Updates soon...
            """
        )
        
    option = st.sidebar.radio('How do you want to input your text?', ('Use an example text', 'Paste a copied', 'Upload a text file'))
    if option == 'Use an example text':
       example_fname = st.sidebar.selectbox('Select example text:', sorted([f for f in os.listdir(EXAMPLES_DIR)
                                                  if f.startswith(('ex'))]))  
       with open(os.path.join(EXAMPLES_DIR, example_fname), 'r', encoding='utf8') as example_file:
           example_text = example_file.read()
           input_text = st.text_area('Analyze the text in the box:', example_text, height=150)
    elif option == 'Upload a text file':
        text = upload_multiple_files()
        # text = uploadfile()
        input_text = st.text_area('Visualize uploaded text:', text, height=150)
    else:
        input_text = st.text_area('Type or paste your text into the text box:', '<Please enter your text...>', height=150)
        
    # side = st.sidebar.selectbox("Select an option below", ["NER",]) # ("Sentiment", "Subjectivity", "NER")

    # nlp = spacy.load('en_core_web_sm')
    # doc = nlp(input_text)
    # st.write(f"Noun phrases: {[chunk.text for chunk in doc.noun_chunks]}")
    # nouns = Counter([token.lemma_ for token in doc if token.pos_ == "NOUN"])
    # verbs = Counter([token.lemma_ for token in doc if token.pos_ == "VERB"])
    # st.write("Nouns:", nouns)
    # st.write("Verbs:", verbs)

    # st.markdown("**Word Cloud**")
    # mask = np.array(Image.open('img/welsh_flag.png'))      
    # maxWords = 20
    # #creating wordcloud
        
    # wordcloud = WordCloud(
        # max_words=maxWords,
        # stopwords=STOPWORDS,
        # width=2000, height=1000,
        # # contour_color= "black", 
        # relative_scaling = 0,
        # mask=mask,
        # background_color="white",
        # font_path='font/Ubuntu-B.ttf'
    # ).generate_from_frequencies(verbs) #.generate(input_text)
    
    # # wordcloud = WordCloud(width = 10, height = 20).generate_from_frequencies(nouns)

    # color = st.radio('Switch image colour:', ('Color', 'Black'))
    # img_cols = ImageColorGenerator(mask) if color == 'Black' else None
    
    # # image_colors = ImageColorGenerator(mask)
    # plt.figure(figsize=[20,15])
    
    # plt.imshow(wordcloud.recolor(color_func=img_cols), interpolation="bilinear")
    # plt.axis("off")
    # st.set_option('deprecation.showPyplotGlobalUse', False)
    # st.pyplot()