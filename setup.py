import os
import re
import string
import spacy
import nltk
import random
import json
import en_core_web_sm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import networkx as nx
from PIL import Image
from io import StringIO
from textblob import TextBlob
# from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import word_tokenize, sent_tokenize, ngrams
from collections import Counter
from keybert import KeyBERT
from summa.summarizer import summarize as summa_summarizer
from wordcloud import WordCloud, ImageColorGenerator
from nltk.corpus import stopwords
nltk.download('punkt') # one time execution
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
random.seed(10)
# For keyword extraction
# For Flair (Keybert) ToDo
# from flair.embeddings import TransformerDocumentEmbeddings



# Update with the Welsh stopwords (source: https://github.com/techiaith/ataleiriau)
en_stopwords = list(stopwords.words('english'))
cy_stopwords = open('welsh_stopwords.txt', 'r', encoding='utf8').read().split('\n')
STOPWORDS = set(en_stopwords + cy_stopwords)
PUNCS = '''!→()-[]{};:'"\,<>./?@#$%^&*_~'''

# EXAMPLES_DIR = 'example_texts_pub'
EXAMPLES_DIR = 'example_texts_cadw'

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

#--------------Get Top n most_common words plus counts---------------
@st.cache
def getTopNWords(text, topn=5, removeStops=False):
    text = text.translate(text.maketrans("", "", string.punctuation))
    text = [word for word in text.lower().split()
                if word not in STOPWORDS] if removeStops else text.lower().split()
    return Counter(text).most_common(topn)        
        
#---------------------keyword in context ----------------------------
@st.cache
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

#------------------------ get collocation ------------------------
@st.cache
def get_collocs(kwic_insts, topn=5, removeStops=False):
    words=[]
    for l, t, r in kwic_insts:
        words += l.split() + r.split()
    all_words = [word for word in words if removeStops and (word not in STOPWORDS)]
    return Counter(all_words).most_common(topn)

#------------------------ plot collocation ------------------------
@st.cache(suppress_st_warning=True)
def plot_collocation(keyword, collocs):
    # st.write(collocs)
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
def gen_ngram(text, _ngrams=2, topn=10):
    if _ngrams==1:
        return getTopNWords(text, topn)
    ngram_list=[]
    for sent in sent_tokenize(text):
        for char in sent:
            if char in PUNCS: sent = sent.replace(char, "")
        ngram_list += ngrams(word_tokenize(sent), _ngrams)
    return [(f"{' '.join(ng):>27s}", c) 
            for ng, c in Counter(ngram_list).most_common(topn)]

#---Polarity score
def get_sentiment(polarity):
  return 'Very Positive' if polarity >= 0.5 else 'Positive' if (
    0.5 > polarity > 0.0) else 'Negative' if (0.0 > polarity >= -0.5
    ) else 'Very Negative' if -0.5 > polarity else 'Neutral'

#---Subjectivity score
@st.cache
def get_subjectivity(subjectivity):
  return 'SUBJECTIVE' if subjectivity > 0.5 else 'OBJECTIVE'
#---Subjectivity distribution
@st.cache
def get_subjectivity_distribution(scores, sentiment_class):
  count = Counter([b for _, _, a, _, b in scores if a==sentiment_class])
  return count['OBJECTIVE'], count['SUBJECTIVE']

@st.cache
def plotfunc(pct, data):
  absolute = int(np.round(pct/100.*np.sum([sum(d) for d in data])))
  return "{:.1f}%\n({:d} reviews)".format(pct, absolute)
# ---------------------
@st.cache
def process_sentiments(text):
  all_reviews = sent_tokenize(text)
  # -------------------
  sentiment_scores = []
  # -------------------
  sentiments_list = []
  subjectivity_list = []

  all_reviews_blob = TextBlob("\n".join(all_reviews))
  # print(f"Reviews Summary:\n- Token + tags: {len(ten_reviews_blob.tags)}")
  print(f"- Words: {len(all_reviews_blob.words)}")
  print(f"- Sentences: {len(all_reviews_blob.sentences)}")
  #-------------------
  for review in all_reviews:
    blob = TextBlob(review)
    polarity, subjectivity = blob.sentiment
    sentiment_class, subjectivity_category = get_sentiment(polarity), get_subjectivity(subjectivity)
    sentiments_list.append(sentiment_class)
    subjectivity_list.append(subjectivity_category)
    sentiment_scores.append((review, polarity, sentiment_class, subjectivity, subjectivity_category))
  # -------------------
  very_positive = get_subjectivity_distribution(sentiment_scores,'Very Positive')
  positive = get_subjectivity_distribution(sentiment_scores,'Positive')
  neutral = get_subjectivity_distribution(sentiment_scores,'Neutral')
  negative = get_subjectivity_distribution(sentiment_scores,'Negative')
  very_negative = get_subjectivity_distribution(sentiment_scores,'Very Negative')
  return sentiment_scores, (very_positive, positive, neutral, negative, very_negative)
# ---------------------

def plot_sentiments(data, fine_grained=True):
  fig, ax = plt.subplots(figsize=(6,6))
  size = 0.5  
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
        pctdistance=0.80, textprops=dict(color="w", weight="bold", size=10))

  # ax.pie(vals.flatten(), radius=1-size, colors=inner_colors,
  #       autopct=lambda pct: plotfunc(pct, data),
  #       wedgeprops=dict(width=size, edgecolor='w'),
  #       pctdistance=0.80, textprops=dict(color="r", size=10))

  ax.set_title("Sentiment Analysis Chart")
  ax.legend(wedges, labels, title="Sentiments", loc="center left", fontsize=12,
            bbox_to_anchor=(1, 0, 0.5, 1))
  # plt.setp(autotexts, size=10, weight="bold")
  # plt.show()
  st.set_option('deprecation.showPyplotGlobalUse', False)
  st.pyplot()

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
       # example_fname = st.sidebar.selectbox('Select example text:', sorted([f for f in os.listdir(EXAMPLES_DIR) if f.startswith(('ex'))]))
       example_fname = st.sidebar.selectbox('Select example text:', sorted([f for f in os.listdir(EXAMPLES_DIR) if f.startswith(('en'))]))
       
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
    
    with col0:
        st.markdown("**NGram Frequency**")
        with st.expander("ℹ️ - Settings", expanded=False):
            if input_text:
                ngrms = st.number_input("Select ngrams",
                    value=2,
                    min_value=1,
                    max_value=5,
                    help='The maximum number of ngrams.'
                    )
                topn = st.number_input("Select top ngrams",
                    value=10,
                    step=5,
                    min_value=5,
                    max_value=50,
                    help='The top (most frequent) ngrams.'
                    )
                col0_lcase = st.checkbox("Lowercase?")
                if col0_lcase: input_text = input_text.lower()

                top_ngrams = gen_ngram(input_text, int(ngrms), int(topn))
                top_ngrams_df = pd.DataFrame(top_ngrams,
                    columns =['NGrams', 'Counts'])
                st.dataframe(top_ngrams_df)

    with col1:
        st.markdown("**Word Cloud**")
        with st.expander("ℹ️ - Settings", expanded=False):
            if input_text:
                mask = np.array(Image.open('img/welsh_flag.png'))      
                # maxWords = st.slider('Maximum number of words:', 10, 300, 300, 10)
                maxWords = st.number_input("Number of words:",
                    value=300,
                    step=50,
                    min_value=50,
                    max_value=300,
                    help='Maximum number of words featured in the cloud.'
                    )
                nlp = spacy.load('en_core_web_sm')
                doc = nlp(input_text)
                
                nouns = Counter([token.lemma_ for token in doc if token.pos_ == "NOUN"])
                verbs = Counter([token.lemma_ for token in doc if token.pos_ == "VERB"])
                proper_nouns = Counter([token.lemma_ for token in doc if token.pos_ == "PROPN"])
                adjectives = Counter([token.lemma_ for token in doc if token.pos_ == "ADJ"])
                adverbs = Counter([token.lemma_ for token in doc if token.pos_ == "ADV"])
                numbers = Counter([token.lemma_ for token in doc if token.pos_ == "NUM"])

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
                
                cloud_type = st.selectbox('Choose cloud type:', ['All words', 'Nouns', 'Proper nouns', 'Verbs', 'Adjectives', 'Adverbs', 'Numbers'])
                if cloud_type == 'All words':
                    wordcloud = wc.generate(input_text)        
                elif cloud_type == 'Nouns':
                    wordcloud = wc.generate_from_frequencies(nouns)        
                elif cloud_type == 'Proper nouns':
                    wordcloud = wc.generate_from_frequencies(proper_nouns)        
                elif cloud_type == 'Verbs':
                    wordcloud = wc.generate_from_frequencies(verbs)
                elif cloud_type == 'Adjectives':
                    wordcloud = wc.generate_from_frequencies(adjectives)
                elif cloud_type == 'Adverbs':
                    wordcloud = wc.generate_from_frequencies(adverbs)
                elif cloud_type == 'Numbers':
                    wordcloud = wc.generate_from_frequencies(numbers)
                else: 
                    pass

                color = st.radio('Switch image colour:', ('Color', 'Black'))
                img_cols = ImageColorGenerator(mask) if color == 'Black' else None
                
                # image_colors = ImageColorGenerator(mask)
                plt.figure(figsize=[20,15])
                
                # plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear")
                plt.imshow(wordcloud.recolor(color_func=img_cols), interpolation="bilinear")
                plt.axis("off")
                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.pyplot()
    
    with col2: #Could you replace with NLTK concordance later?
        st.markdown("**Keyword in Context**")
        with st.expander("ℹ️ - Settings", expanded=False):
            if input_text:
                topwords = [f"{w} ({c})" for w, c in getTopNWords(input_text, removeStops=True)]
                # st.write(True if topwords else False)
                keyword = st.selectbox('Select a keyword:', topwords).split('(',1)[0].strip()
                window_size = st.slider('Select the window size:', 1, 10, 2)
                maxInsts = st.slider('Maximum number of instances:', 5, 50, 10, 5)
                col2_lcase = st.checkbox("Lowercase?", key='col2_checkbox')
                kwic_instances = get_kwic(input_text, keyword, window_size, maxInsts, col2_lcase)

                keyword_analysis = st.radio('Anaysis:', ('Keyword in context', 'Collocation'))
                if keyword_analysis == 'Keyword in context':
                    kwic_instances_df = pd.DataFrame(kwic_instances,
                        columns =['left context', 'keyword', 'right context'])
                    st.dataframe(kwic_instances_df)
                else: #Could you replace with NLTK concordance later? 
                    # keyword = st.text_input('Enter a keyword:','staff')
                    collocs = get_collocs(kwic_instances) #TODO: Modify to accept 'topn'
                    colloc_str = ', '.join([f"{w}[{c}]" for w, c in collocs])
                    st.write(f"Collocations for '{keyword}':\n{colloc_str}")
                    plot_collocation(keyword, collocs)

# @st.cache(suppress_st_warning=True)
def run_sentiments():
    with st.expander("ℹ️ - About Sentiment Analyzer", expanded=False):
        st.markdown(
            """
            ToDo: Describe the sentiment analyzer...
            """
        )
    # st.markdown('### 🔍 Visualization')
    option = st.sidebar.radio('How do you want to input your text?', ('Use an example text', 'Paste copied text', 'Upload files'))
    if option == 'Use an example text':
       example_fname = st.sidebar.selectbox('Select example text:', sorted([f for f in os.listdir(EXAMPLES_DIR) if f.startswith(('en'))]))
       
       with open(os.path.join(EXAMPLES_DIR, example_fname), 'r', encoding='utf8') as example_file:
           example_text = example_file.read()
           input_text = st.text_area('Visualize example text in the box:', example_text, height=150)
    elif option == 'Upload files':
        text = upload_multiple_files()
        input_text = st.text_area('Visualize uploaded text:', text, height=150)
    else:
        input_text = st.text_area('Type or paste your text into the text box:', '<Please enter your text...>', height=150)
    
    col1, col2 = st.columns(2)
    with col1:
        option = st.radio('How do you want to categorize the sentiments?', ('3 Class Sentiments', '5 Class Sentiments'))
        data = process_sentiments(input_text)
        if option == '3 Class Sentiments':
            plot_sentiments(data[1], fine_grained=False)
        else:
            plot_sentiments(data[1])

    with col2:
        num_examples = st.slider('Number of example [5 to 20%]',  min_value=5, max_value=20, step=5)
        df = pd.DataFrame(data[0], columns =['Review','Polarity', 'Sentiment', 'Subjectivity', 'Category'])
        df = df[['Review','Polarity', 'Sentiment']]
        st.dataframe(df.head(num_examples))

def run_keyphrase():

# --------------------- Keyword/KeyPhrase ------------------------------
# Borrowed from https://github.com/streamlit/example-app-bert-keyword-extractor
    st.markdown("**Keyword/KeyPhrase Extraction**")
    # language = st.sidebar.selectbox('Newid iaith (Change language):', ['Cymraeg', 'English'])
    with st.expander("ℹ️ - About this Keyword/KeyPhrase Extract", expanded=False):
        st.write(
            """     
            -   The *BERT Keyword Extractor* app is an easy-to-use interface built in Streamlit for the amazing [KeyBERT](https://github.com/MaartenGr/KeyBERT) library from Maarten Grootendorst!
            -   It uses a minimal keyword extraction technique that leverages multiple NLP embeddings and relies on [Transformers] (https://huggingface.co/transformers/) 🤗 to create keywords/keyphrases that are most similar to a document.
            """
        )

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

    ce, c1, ce, c2, ce = st.columns([0.05, 2, 0.05, 5, 0.07])
    with c1:
        model_type = st.radio( "Choose your model", ["DistilBERT (Default)", "Flair"],
        help="Only the DistilBERT works for now!",
    )
        if model_type == "Default (DistilBERT)":
        # kw_model = KeyBERT(model=roberta)
            @st.cache(allow_output_mutation=True)
            def load_model():
                return KeyBERT(model=roberta)
            kw_model = load_model()
        else:
            @st.cache(allow_output_mutation=True)
            def load_model():
                return KeyBERT("distilbert-base-nli-mean-tokens")
            kw_model = load_model()

        top_N = st.slider(
            "# of results",
            min_value=1,
            max_value=30,
            value=10,
            help="You can choose the number of keywords/keyphrases to display. Between 1 and 30, default number is 10.",
        )
        min_Ngrams = st.number_input(
            "Minimum Ngram",
            min_value=1,
            max_value=4,
            help="The minimum value for the ngram range. *Keyphrase_ngram_range* sets the length of the resulting keywords/keyphrases. To extract keyphrases, simply set *keyphrase_ngram_range* to (1, 2) or higher depending on the number of words you would like in the resulting keyphrases."
        )

        max_Ngrams = st.number_input(
            "Maximum Ngram",
            value=2,
            min_value=1,
            max_value=4,
            help="The maximum value for the keyphrase_ngram_range. *Keyphrase_ngram_range* sets the length of the resulting keywords/keyphrases. To extract keyphrases, simply set *keyphrase_ngram_range* to (1, 2) or higher depending on the number of words you would like in the resulting keyphrases."
        )

        StopWordsCheckbox = st.checkbox(
            "Remove stop words",
            help="Tick this box to remove stop words from the document (currently English only)",
        )

        use_MMR = st.checkbox(
            "Use MMR",
            value=True,
            help="You can use Maximal Margin Relevance (MMR) to diversify the results. It creates keywords/keyphrases based on cosine similarity. Try high/low 'Diversity' settings below for interesting variations.",
        )

        Diversity = st.slider(
            "Keyword diversity (MMR only)",
            value=0.5,
            min_value=0.0,
            max_value=1.0,
            step=0.1,
            help="The higher the setting, the more diverse the keywords. Note that the *Keyword diversity* slider only works if the *MMR* checkbox is ticked."
        )
    with c2:
        MAX_WORDS = 1000
        res = len(re.findall(r"\w+", input_text))
        if res > MAX_WORDS:
            st.warning(
                "⚠️ Your text contains "
                + str(res)
                + " words."
                + " Only the first " + res + " words will be reviewed. Stay tuned as increased allowance is coming! 😊"
            )

        doc = input_text[:MAX_WORDS]
        # submit_button = st.form_submit_button(label="✨ Get me the data!")

        mmr = True if use_MMR else False

        StopWords = "english" if StopWordsCheckbox else None

        if min_Ngrams > max_Ngrams:
            st.warning("min_Ngrams can't be greater than max_Ngrams")

        keywords = kw_model.extract_keywords(
            doc,
            keyphrase_ngram_range=(min_Ngrams, max_Ngrams),
            use_mmr = mmr,
            stop_words = StopWords,
            top_n = top_N,
            diversity= Diversity
        )

        df = (
            pd.DataFrame(keywords, columns=["Keyword/Keyphrase", "Relevancy"])
            .sort_values(by="Relevancy", ascending=False)
            .reset_index(drop=True)
        )

        df.index += 1

        # Add styling
        cmGreen = sns.light_palette("green", as_cmap=True)
        cmRed = sns.light_palette("red", as_cmap=True)
        df = df.style.background_gradient(
            cmap=cmGreen,
            subset=[
                "Relevancy",
            ],
        )

        format_dictionary = {
            "Relevancy": "{:.1%}",
        }

        df = df.format(format_dictionary)
        st.table(df)

