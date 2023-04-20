def render_d3_collocations(collocations_data):
    # Download index.html from GitHub
    index_html_url = "https://raw.githubusercontent.com/UCREL/welsh-freetxt-app/main/d3_collocations/index.html"
    index_html_content = requests.get(index_html_url).text

    # Insert collocationsData variable into index_html_content
    index_html_content = index_html_content.replace(
        "<div id=\"graph\"></div>",
        f"<div id=\"graph\"></div><script>const collocationsData = {json.dumps(collocations_data)};</script>"
    )

    # Set up the output
    d3_component = components.declare_component(
        "d3_component",
        url="https://d3-component.herokuapp.com",
    )

    # Call the component with the modified index.html content
    return d3_component(
        index_html_content,
        collocations_data=collocations_data,
        key="d3_collocations",
        height=800,
        width=800,
    )
