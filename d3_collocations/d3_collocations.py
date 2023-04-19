import os
import streamlit.components.v1 as components
import requests

def render_d3_collocations(collocations_data):
    # Download index.html from GitHub
    index_html_url = "https://raw.githubusercontent.com/UCREL/welsh-freetxt-app/edit/main/d3_collocations/index.html"
    index_html_content = requests.get(index_html_url).text
    
    return components.html(
        index_html_content,
        collocations_data=collocations_data,
        height=800,
        width=800,
    )


