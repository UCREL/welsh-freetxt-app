import os
import string
import random
import streamlit as st
import base64
from PIL import Image
from labels import MESSAGES
from streamlit_lottie import st_lottie

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

st.markdown("# FreeTxt Text Analysis")

        
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
        st.subheader("[Reviews analysis and illustrations](https://ucrel-welsh-freetxt-app-home-i3gq4l.streamlit.app/Reviews_analysis_&_illustrations)")
        st.write('''This is one of the key features of the tool and has three core components:
a. Data View: This allows the user to display and visualize the selected columns from the data file they wish to look at. The user can also dynamically modify the selection or the order of the columns as they wish before performing any other task on the selected columns
b. Word Cloud: This creates a word cloud from the content of the selected columns. It also allows the user to select the column(s) to build the word cloud from as well as the word cloud type – i.e. 'All words', 'Bigrams', 'Trigrams', '4-grams', 'Nouns', 'Proper nouns', 'Verbs', 'Adjectives', 'Adverbs', 'Numbers'
c. Key word in Context and Collocation: This extracts the keywords in the review text from the selected columns as well as the contexts within which they appeared in the text allowing the user to adjust the context window. It also shows the collocated words with the selected keywords''')

       
    with right_column:
        data_url = read_gif("img/visualization.gif")
        st.markdown(
            f'<p style="text-align: center; color: grey;"><a href="https://ucrel-welsh-freetxt-app-home-i3gq4l.streamlit.app/Reviews_analysis_&_illustrations"><img width="200" height="200" src="data:image/gif;base64,{data_url} "></a></p>',
            unsafe_allow_html=True,  
                )
st.write("---")
with st.container():
    
    left_column, right_column = st.columns([1, 1])
    with left_column:
        data_url_2 = read_gif("img/reviews.gif")
        st.markdown(
            f'<p style="text-align: center; color: grey;"><a href="https://ucrel-welsh-freetxt-app-home-i3gq4l.streamlit.app/Positive_and_Negative_reviews"><img width="200" height="200" src="data:image/gif;base64,{data_url_2} "></a></p>',
            unsafe_allow_html=True, 
                )
        
    with right_column:
        st.subheader("[Positive and Negative reviews](https://ucrel-welsh-freetxt-app-home-i3gq4l.streamlit.app/Positive_and_Negative_reviews)")
        st.write("This feature performs sentiment classification on reviews from selected column(s) and displays a pie chart to visualize the output") 
      
        
#######################
st.write("---")
with st.container():
    
    left_column, right_column = st.columns([1, 1])
    with left_column:
        st.subheader("[Generate_a_summary](https://ucrel-welsh-freetxt-app-home-i3gq4l.streamlit.app/Generate_a_summary)")
        st.write('This tool, adapted from the Welsh Summarization project, produces a basic extractive summary of the review text from the selected columns.')
    with right_column:
        data_url = read_gif("img/summary.gif")
        st.markdown(
            f'<p style="text-align: center; color: grey;"><a href="https://ucrel-welsh-freetxt-app-home-i3gq4l.streamlit.app/Generate_a_summary"><img width="400" height="400" src="data:image/gif;base64,{data_url} "></a></p>',
            unsafe_allow_html=True,  
               )
st.write("---")
with st.container():
    
    left_column, right_column = st.columns([1, 1])
    with left_column:
        data_url_2 = read_gif("img/semantic.gif")
        st.markdown(
            f'<p style="text-align: center; color: grey;"><a href="https://ucrel-welsh-freetxt-app-home-i3gq4l.streamlit.app/Word_Types_and_Relations"><img width="500" height="500" src="data:image/gif;base64,{data_url_2} "></a></p>',
            unsafe_allow_html=True, 
                )

    with right_column:
        st.subheader("[Word_Types_and_Relations](https://ucrel-welsh-freetxt-app-home-i3gq4l.streamlit.app/Word_Types_and_Relations)")
        st.write('''This feature uses the PyMUSAS pipeline on Spacy to generate and display POS (CyTag) tags as well as semantic (USAS) tags. 
						It currently works on the Ucrel-freetxt-VM as setting up Docker on the Streamlit cloud is a bit complex''')

        text = "Sefydliad cyllidol yw bancwr neu fanc sy'n actio fel asiant talu ar gyfer cwsmeriaid, ac yn rhoi benthyg ac yn benthyg arian. Yn rhai gwledydd, megis yr Almaen a Siapan, mae banciau'n brif berchenogion corfforaethau diwydiannol, tra mewn gwledydd eraill, megis yr Unol Daleithiau, mae banciau'n cael eu gwahardd rhag bod yn berchen ar gwmniau sydd ddim yn rhai cyllidol. Adran Iechyd Cymru."
  
        
        










# ---- HIDE STREAMLIT STYLE ----
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)
