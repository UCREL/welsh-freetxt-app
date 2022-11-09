import streamlit as st
import base64
from PIL import Image
from labels import MESSAGES



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
add_logo("img/FreeTxt_logo.png") 
st.markdown("# Reviews analysis & illustrations")
st.write('''This is one of the key features of the tool and has three core components:

a. Data View: This allows the user to display and visualize the selected columns from the data file they wish to look at. The user can also dynamically modify the selection or the order of the columns as they wish before performing any other task on the selected columns

b. Word Cloud: This creates a word cloud from the content of the selected columns. It also allows the user to select the column(s) to build the word cloud from as well as the word cloud type – i.e. 'All words', 'Bigrams', 'Trigrams', '4-grams', 'Nouns', 'Proper nouns', 'Verbs', 'Adjectives', 'Adverbs', 'Numbers'

c. Key word in Context and Collocation: This extracts the keywords in the review text from the selected columns as well as the contexts within which they appeared in the text allowing the user to adjust the context window. It also shows the collocated words with the selected keywords''')
