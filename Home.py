import os
import string
import random
import streamlit as st
import base64
from PIL import Image
from labels import MESSAGES
from streamlit_lottie import st_lottie
##multilingual
import gettext
_ = gettext.gettext


#################################################################################

# Use local CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

#local_css("style/style.css")

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
     page_title='Welsh FreeTxt Tool',
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
language = st.sidebar.selectbox('', ['en', 'cy'])
try:
  localizator = gettext.translation('base', localedir='locales', languages=[language])
  localizator.install()
  _ = localizator.gettext 
except:
    pass
st.markdown (_("# FreeTxt Text Analysis"))

        
add_logo("img/FreeTxt_logo.png") 
st.write("---")


######### gif from local file"""

def read_gif(name):

    file_ = open(name, "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()
    return data_url

#######################
with st.container():
    
    left_column, right_column = st.columns([1, 1])
    with left_column:
        st.subheader(_("[Reviews analysis and illustrations](http://ucrel-freetxt-1.lancs.ac.uk:8080/Reviews_analysis_and_illustrations)"))
        st.write(_('''This tool has three components: 
                1. Data View: to select, view and filter columns from a data file  
               2. Word Cloud: creates a word cloud from content in the selected columns of a file 
                3. Context and Collocation: extracts the most frequent words that appear in the selected columns of your file, illustrating how they appear in sentences. It also shows the words which most often co-occur with these most frequent words'''))
       
    with right_column:
        data_url = read_gif("img/visualization.gif")
        st.markdown(
            f'<p style="text-align: center; color: grey;"><a href="http://ucrel-freetxt-1.lancs.ac.uk:8080/Reviews_analysis_and_illustrations"><img width="200" height="200" src="data:image/gif;base64,{data_url} "></a></p>',
            unsafe_allow_html=True,  
                )
st.write("---")
with st.container():
    
    left_column, right_column = st.columns([1, 1])
    with left_column:
        data_url_2 = read_gif("img/reviews.gif")
        st.markdown(
            f'<p style="text-align: center; color: grey;"><a href="http://ucrel-freetxt-1.lancs.ac.uk:8080/Positive_and_Negative_reviews"><img width="200" height="200" src="data:image/gif;base64,{data_url_2} "></a></p>',
            unsafe_allow_html=True, 
                )
        
    with right_column:
        st.subheader(_("[Positive and Negative reviews](http://ucrel-freetxt-1.lancs.ac.uk:8080/Positive_and_Negative_reviews)"))
        st.write(_("This feature performs sentiment classification on reviews from selected column(s) and displays a pie chart to visualize the output")) 
      
        
#######################
st.write("---")
with st.container():
    
    left_column, right_column = st.columns([1, 1])
    with left_column:
        st.subheader(_("[Generate_a_summary](http://ucrel-freetxt-1.lancs.ac.uk:8080/Generate_a_summary)"))
        st.write(_('This tool, adapted from the Welsh Summarization project, produces a basic extractive summary of the review text from the selected columns.'))
    with right_column:
        data_url = read_gif("img/summary.gif")
        st.markdown(
            f'<p style="text-align: center; color: grey;"><a href="http://ucrel-freetxt-1.lancs.ac.uk:8080/Generate_a_summary"><img width="400" height="400" src="data:image/gif;base64,{data_url} "></a></p>',
            unsafe_allow_html=True,  
               )
st.write("---")
with st.container():
    
    left_column, right_column = st.columns([1, 1])
    with left_column:
        data_url_2 = read_gif("img/semantic.gif")
        st.markdown(
            f'<p style="text-align: center; color: grey;"><a href="http://ucrel-freetxt-1.lancs.ac.uk:8080/Word_Types_and_Relations"><img width="500" height="500" src="data:image/gif;base64,{data_url_2} "></a></p>',
            unsafe_allow_html=True, 
                )

    with right_column:
        st.subheader(_("[Word_Types_and_Relations](http://ucrel-freetxt-1.lancs.ac.uk:8080/Word_Types_and_Relations)"))
        st.write(_('This feature uses the PyMUSAS pipeline on Spacy to generate and display POS (CyTag) tags as well as semantic (USAS) tags. It currently works on the Ucrel-freetxt-VM as setting up Docker on the Streamlit cloud is a bit complex'))

        text = "Sefydliad cyllidol yw bancwr neu fanc sy'n actio fel asiant talu ar gyfer cwsmeriaid, ac yn rhoi benthyg ac yn benthyg arian. Yn rhai gwledydd, megis yr Almaen a Siapan, mae banciau'n brif berchenogion corfforaethau diwydiannol, tra mewn gwledydd eraill, megis yr Unol Daleithiau, mae banciau'n cael eu gwahardd rhag bod yn berchen ar gwmniau sydd ddim yn rhai cyllidol. Adran Iechyd Cymru."
  
        
        









# ---- HIDE STREAMLIT STYLE ----
#hide_st_style = """
          #  <style>
            #MainMenu {visibility: hidden;}
        #    footer {visibility: hidden;}
         #   #header {visibility: hidden;}
          #  </style>
           # """
#st.markdown(hide_st_style, unsafe_allow_html=True)
