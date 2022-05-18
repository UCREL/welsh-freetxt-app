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

task = st.sidebar.radio("Select a task", ('🌷 Summarization','Visualization', 'N-Gram Frequency Counting', 'Keyword in Context',
                                          'Part of Speech Tagging', 'Semantic Tagging', 'Machine Translation', 'Sentiment Analysis'))

if task == 'Summarization':
    run_text_summarizer()
elif task == 'Visualization':
    run_visualizer()
else:
    st.write(task, "is under construction...")