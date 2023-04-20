
def render_d3_collocations(collocations_data):
    # Download index.html from GitHub
    index_html_url = "https://raw.githubusercontent.com/UCREL/welsh-freetxt-app/main/d3_collocations/index.html"
    index_html_content = requests.get(index_html_url).text

    # Convert collocations_data to a JSON string
    collocations_data_json = json.dumps(collocations_data)
    
    # Replace the collocations_data_placeholder with the actual data
    index_html_content = index_html_content.replace("<!--collocations_data_placeholder-->", json.dumps(collocations_data))

    # Return the HTML content as a string
    return index_html_content
