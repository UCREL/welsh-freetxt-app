st.set_page_config(
     page_title='Adnodd Creu Crynodebau (ACC)',
     page_icon='🏴󠁧󠁢󠁷󠁬󠁳󠁿',
     layout="wide",
     initial_sidebar_state="expanded",
     menu_items={
         'Get Help': "https://wp.lancs.ac.uk/acc/",
         'Report a bug': "https://wp.lancs.ac.uk/acc/",
         'About': '''## The Welsh FreeTxt tool!'''
     }
 )

language = st.sidebar.selectbox('Newid iaith (Change language):', ['Cymraeg', 'English'])
st.header('🏴󠁧󠁢󠁷󠁬󠁳󠁿 The Welsh FreeTxt tool!')
st.subheader('Enter your text below:')

option = st.radio(
  'How do you want to input your text?',
  ('Use example text', 'Enter your own text'))

chosen_ratio = st.sidebar.slider('Select summary ratio [10% to 50%]',
        min_value=10, max_value=50, step=10)/100

if option == 'Use example text':
  input_text = st.text_area('Summarise the example text in the box:', example_text)
else:
  input_text = st.text_area('Type or paste your text into the text box:', '<Please enter your text...>')

if st.button("Annotate 👈"):
    st.write('Feature under contruction...')
    
#     if input_text and input_text!='<Please enter your text...>':
#      summary = text_rank_summarize(input_text, ratio=chosen_ratio)
#      if summary:
#         st.write(text_rank_summarize(input_text, ratio=chosen_ratio))
#      else:
#         st.write(sent_tokenize(text_rank_summarize(input_text, ratio=0.5))[0])
#      # process what needs to be displayed with regards to ratio
#     else:
#      st.write('Please enter your text')
