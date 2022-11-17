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
        data_url = read_gif("img/visualization.gif")
        st.markdown(
            f'<p style="text-align: center; color: grey;"><a href="https://ucrel-welsh-freetxt-app-home-i3gq4l.streamlit.app/Reviews_analysis_&_illustrations"><img width="200" height="200" src="data:image/gif;base64,{data_url} "></a></p>',
            unsafe_allow_html=True,  
                )
    with right_column:
        st.subheader("[Positive and Negative reviews](https://ucrel-welsh-freetxt-app-home-i3gq4l.streamlit.app/Positive_and_Negative_reviews)")
           
        data_url_2 = read_gif("img/reviews.gif")
        st.markdown(
            f'<p style="text-align: center; color: grey;"><a href="https://ucrel-welsh-freetxt-app-home-i3gq4l.streamlit.app/Positive_and_Negative_reviews"><img width="200" height="200" src="data:image/gif;base64,{data_url_2} "></a></p>',
            unsafe_allow_html=True, 
                )
#######################
st.write("---")
with st.container():
    
    left_column, right_column = st.columns([1, 1])
    with left_column:
        st.subheader("[Generate_a_summary](https://ucrel-welsh-freetxt-app-home-i3gq4l.streamlit.app/Generate_a_summary)")
        data_url = read_gif("img/summary.gif")
        st.markdown(
            f'<p style="text-align: center; color: grey;"><a href="https://ucrel-welsh-freetxt-app-home-i3gq4l.streamlit.app/Generate_a_summary"><img width="400" height="400" src="data:image/gif;base64,{data_url} "></a></p>',
            unsafe_allow_html=True,  
                )
    with right_column:
        st.subheader("[Word_Types_and_Relations](https://ucrel-welsh-freetxt-app-home-i3gq4l.streamlit.app/Word_Types_and_Relations)")
           
        data_url_2 = read_gif("img/semantic.gif")
        st.markdown(
            f'<p style="text-align: center; color: grey;"><a href="https://ucrel-welsh-freetxt-app-home-i3gq4l.streamlit.app/Word_Types_and_Relations"><img width="500" height="500" src="data:image/gif;base64,{data_url_2} "></a></p>',
            unsafe_allow_html=True, 
                )

        










# ---- HIDE STREAMLIT STYLE ----
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)
