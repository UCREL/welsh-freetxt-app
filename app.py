from setup import *
st.set_page_config(
     page_title='The Welsh Free Text Tool',
     page_icon='🌼',
     layout="wide",
     initial_sidebar_state="expanded",
     menu_items={
         'Get Help': "https://wp.lancs.ac.uk/acc/",
         'Report a bug': "https://wp.lancs.ac.uk/acc/",
         'About': '''## Welsh FreeTxt Tool!'''
     }
 )

st.sidebar.header('🌼 The Welsh FreeTxt Tool')

task = st.sidebar.radio("Select a task", ('Text Visualization', 'N-Gram Frequency Counting', 'Keyword in Context',
                                          'Part of Speech Tagging', 'Semantic Tagging', 'Text Summarization',
                                          'Machine Translation', 'Sentiment Analysis'))
if task == 'Text Summarization':
     language = st.sidebar.selectbox('Newid iaith (Change language):', ['Cymraeg', 'English'])
     if language=='Cymraeg':
          st.subheader("🌷 Croeso i’r Adnodd Creu Crynodebau (ACC) f.1.0")

          option = st.radio(
               'Sut ydych chi am fewnbynnu eich testun?',
               ('Defnyddiwch destun enghreifftiol', 'Rhowch eich testun eich hun'))

          chosen_ratio = st.sidebar.slider('Dewiswch gymhareb y crynodeb [10% i 50%]:',
                     min_value=10, max_value=50, step=10)/100

          if option == 'Defnyddiwch destun enghreifftiol':
               input_text = st.text_area('Crynhowch y testun enghreifftiol yn y blwch:', example_text)
          else:
               input_text = st.text_area('Teipiwch neu gludwch eich testun yn y blwch testun', '<Rhowch eich testun...>')

          if st.button("Crynhoi👈"):
            if input_text and input_text!='<Rhowch eich testun (Please enter your text...)>':
              summary = text_rank_summarize(input_text, ratio=chosen_ratio)
              if summary:
                 st.write(text_rank_summarize(input_text, ratio=chosen_ratio))
              else:
                 st.write(sent_tokenize(text_rank_summarize(input_text, ratio=0.5))[0])
            else:
              st.write("Rhowch eich testun...(Please enter your text...)")
     else:
          st.subheader('🌷 Welcome to Welsh Text Summary Creator (ACC) v.1.0')
          option = st.radio(
               'How do you want to input your text?',
               ('Use example text', 'Enter your own text'))

          chosen_ratio = st.sidebar.slider('Select summary ratio [10% to 50%]',
                     min_value=10, max_value=50, step=10)/100

          if option == 'Use example text':
               input_text = st.text_area('Summarise the example text in the box:', example_text)
          else:
               input_text = st.text_area('Type or paste your text into the text box:', '<Please enter your text...>')

          if st.button("Summarise👈"):
            if input_text and input_text!='<Please enter your text...>':
              summary = text_rank_summarize(input_text, ratio=chosen_ratio)
              if summary:
                 st.write(text_rank_summarize(input_text, ratio=chosen_ratio))
              else:
                 st.write(sent_tokenize(text_rank_summarize(input_text, ratio=0.5))[0])
              # process what needs to be displayed with regards to ratio
            else:
              st.write('Please enter your text')
else:
     st.write(task, "is under construction...")
