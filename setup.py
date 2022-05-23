import os
import string
import streamlit_wordcloud as wordcloud
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import nltk
import networkx as nx
from io import StringIO
from nltk.tokenize import sent_tokenize
from summa.summarizer import summarize as summa_summarizer
from wordcloud import WordCloud, STOPWORDS
nltk.download('punkt') # one time execution


#📃📌📈📈📉⛱🏓🏆🎲

## Define summarizer models
# text_rank
def text_rank_summarize(article, ratio):
  return summa_summarizer(article, ratio=ratio)


#helper functions------------------------------------------------------------------

#--------uploading file ---------------
def uploadfile():
    uploaded_file = st.file_uploader("Choose a text file")
    if uploaded_file is not None:
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        return stringio.read()
    else:
        return '<Please upload your file ...>'


#----------keyword in context--------------
def get_kwic(text, keyword, window_size=1, maxInstances=10, lower_case=False):
    text = text.translate(text.maketrans("", "", string.punctuation))
    if lower_case:
        text = text.lower()
        keyword = keyword.lower()
    kwic_insts = []
    tokens = text.split()
    keyword_indexes = [i for i in range(len(tokens)) if tokens[i]==keyword]
    for index in keyword_indexes[:maxInstances]:
        left_context = ' '.join(tokens[index-window_size:index])
        target_word = tokens[index]
        right_context = ' '.join(tokens[index+1:index+window_size+1])
        kwic_insts.append((left_context, target_word, right_context))
    return kwic_insts
    
#apps------------------------------------------------------------------
def run_text_summarizer():
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
           example_fname = st.sidebar.selectbox('Select example text:', ['cy_ex_0_Dulyn', 'cy_ex_1_Menter Iaith Môn', 'cy_ex_2_Pencampwriaeth', 'cy_ex_3_Paris',
           'cy_ex_4_Neuadd y Ddinas', 'cy_ex_5_Y_Gofid_Mawr_Covid19'])

           with open(os.path.join('example_texts', example_fname), 'r', encoding='utf8') as example_file:
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
        st.markdown('## 🌷 Welsh Summary Creator')
        st.markdown("### Enter your text below:")
        option = st.sidebar.radio('How do you want to input your text?', ('Use an example text', 'Paste a copied', 'Upload a text file'))
        if option == 'Use an example text':           
           example_fname = st.sidebar.selectbox('Select example text:', ['en_ex_0_Castell Coch', 'en_ex_1_Beaumaris Castle', 'en_ex_2_Blaenavon Ironworks', 'en_ex_3_Caerleon Roman Baths',
           'en_ex_4_Caernarfon Castle', 'en_ex_5_Caerphilly Castle'])
           with open(os.path.join('example_texts', example_fname), 'r', encoding='utf8') as example_file:
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
            The `Visualizer` tool provides the following features: 
            * Keyword in Context (KWIC):
              - **input**: `text`, `keyword`, `window_size:(default=1)`, `maxInstances=(default=10)`, `lower_case=(False)`
              - **Output**: *list of tuples:* (`left_context`, `keyword`, `right_context`)
            """
        )

    st.markdown('### 🔍 Visualization')
    option = st.sidebar.radio('How do you want to input your text?', ('Use an example text', 'Paste a copied', 'Upload a text file'))
    if option == 'Use an example text':
       # example_fname = st.sidebar.selectbox('Select example text:', ['cy_ex_0_Dulyn', 'cy_ex_1_Menter Iaith Môn', 'cy_ex_2_Pencampwriaeth', 'cy_ex_3_Paris',
       # 'cy_ex_4_Neuadd y Ddinas', 'cy_ex_5_Y_Gofid_Mawr_Covid19'])
       
       example_fname = st.sidebar.selectbox('Select example text:', ['en_ex_0_Castell Coch', 'en_ex_1_Beaumaris Castle', 'en_ex_2_Blaenavon Ironworks', 'en_ex_3_Caerleon Roman Baths',
       'en_ex_4_Caernarfon Castle', 'en_ex_5_Caerphilly Castle'])
       
       
       with open(os.path.join('example_texts', example_fname), 'r', encoding='utf8') as example_file:
           example_text = example_file.read()
           input_text = st.text_area('Visualize example text in the box:', example_text, height=300)
    elif option == 'Upload a text file':
        text = uploadfile()
        input_text = st.text_area('Visualize uploaded text:', text, height=300)
    else:
        input_text = st.text_area('Type or paste your text into the text box:', '<Please enter your text...>', height=300)

    col1, col2 = st.columns(2)
    col1.subheader("Keyword in Context") 
    with col1.form("form1"):
        keyword = st.text_input('Enter a keyword:')
        window_size = st.slider('Select the window size:', 1, 10, 2)
        maxInsts = st.slider('Maximum number of instances:', 5, 50, 10, 5)
        lcase = st.checkbox("Lowercase?")

        # Every form must have a submit button.
        submitted = st.form_submit_button("Visualize 👈")
        if submitted:
            kwic_instances = get_kwic(input_text, keyword, window_size, maxInsts, lcase)
            kwic_instances_df = pd.DataFrame(kwic_instances,
                columns =['left context', 'keyword', 'right context'])
            st.dataframe(kwic_instances_df)
            
    w_cloud = WordCloud(width = 300, height = 200, random_state=1, 
        collocations=False, stopwords = STOPWORDS).generate(input_text)

    col2.subheader("Word Cloud")
    arr = np.random.normal(1, 1, size=100)
    fig, ax = plt.subplots()

    words = [
        dict(text="Robinhood", value=16000, color="#b5de2b", country="US", industry="Cryptocurrency"),
        dict(text="Personio", value=8500, color="#b5de2b", country="DE", industry="Human Resources"),
        dict(text="Boohoo", value=6700, color="#b5de2b", country="UK", industry="Beauty"),
        dict(text="Deliveroo", value=13400, color="#b5de2b", country="UK", industry="Delivery"),
        dict(text="SumUp", value=8300, color="#b5de2b", country="UK", industry="Credit Cards"),
        dict(text="CureVac", value=12400, color="#b5de2b", country="DE", industry="BioPharma"),
        dict(text="Deezer", value=10300, color="#b5de2b", country="FR", industry="Music Streaming"),
        dict(text="Eurazeo", value=31, color="#b5de2b", country="FR", industry="Asset Management"),
        dict(text="Drift", value=6000, color="#b5de2b", country="US", industry="Marketing Automation"),
        dict(text="Twitch", value=4500, color="#b5de2b", country="US", industry="Social Media"),
        dict(text="Plaid", value=5600, color="#b5de2b", country="US", industry="FinTech"),
    ]
    
    col2.wordcloud.visualize(words, tooltip_data_fields={
        'text':'Company', 'value':'Mentions', 'country':'Country of Origin', 'industry':'Industry'
    }, per_word_coloring=False)
    
    
    # ax.hist(arr, bins=20)
    # col2.pyplot(w_cloud)

    # #Set figure size
    # plt.figure(figsize=(40, 30))
    # # Display image
    # plt.imshow(w_cloud) 
    # # No axis 
    # plt.axis("off")
    # col2.plt.show(w_cloud)