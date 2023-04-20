import requests
import json
import streamlit.components.v1 as components


def render_d3_collocations(collocations_data):
    # Download index.html from GitHub
    index_html_url = "https://raw.githubusercontent.com/UCREL/welsh-freetxt-app/main/d3_collocations/index.html"
    index_html_content = requests.get(index_html_url).text

    # Convert collocations_data to a JSON string
    collocations_data_json = json.dumps(collocations_data)

    # Insert collocations_data_json variable into index_html_content
    index_html_content = index_html_content.replace(
        "<!-- collocations_data_placeholder -->",
        collocations_data_json
    )

    # Insert JavaScript code that uses collocationsData variable
    index_html_content += f"""
        <script>
            const collocationsData = {collocations_data_json};
            // D3.js code here to create the visualization using collocationsData
        </script>
    """

    return index_html_content

