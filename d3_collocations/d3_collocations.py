import streamlit.components.v1 as components
import requests
import json

def render_d3_collocations(collocations_data):
    # Download index.html from GitHub
    index_html_url = "https://raw.githubusercontent.com/UCREL/welsh-freetxt-app/main/d3_collocations/index.html"
    index_html_content = requests.get(index_html_url).text

    # Embed collocations_data as a JSON string within the HTML content
    collocations_data_json = json.dumps(collocations_data)
    index_html_content = index_html_content.replace("<!-- collocations_data_placeholder -->", collocations_data_json)

    return components.html(
        index_html_content,
        height=800,
        width=800,
    )
