import streamlit.components.v1 as components

def d3_collocations(collocations_data):
    return components.html(
        open("index.html").read(),
        collocations_data=collocations_data,
        height=800,
        width=800,
    )
