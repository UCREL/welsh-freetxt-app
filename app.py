from setup import *

st.set_page_config(
     page_title='Welsh Free Text Tool',
     page_icon='🌼',
     layout="wide",
     initial_sidebar_state="expanded",
     menu_items={
         'Get Help': "https://ucrel.lancs.ac.uk/freetxt/",
         'Report a bug': "https://github.com/UCREL/welsh-freetxt-app/issues",
         'About': '''## The FreeTxt tool supports bilingual (English and Welsh) free text data analysis of surveys and questionnaire responses'''
     }
 )

st.sidebar.markdown('# 🌼 Welsh FreeTxt')

#📃📌📈📈📉⛱🏓🏆🎲 

task = st.sidebar.radio("Select a task", ('🔍 Visualizer', '📃 Summarizer', '🎲 Sentiment Analyzer')) #, '📉 Analyzer', '📌 Annotator', '📉 Keyphrase Extractor',))

if task == '🔍 Visualizer':
    run_visualizer()
elif task == '📃 Summarizer':
    run_summarizer()
elif task == '🎲 Sentiment Analyzer':
    run_sentiments()
else:
    st.write(task, 'is under construction...')

# elif task == '📉 Keyphrase Extractor':
    # run_keyphrase()
