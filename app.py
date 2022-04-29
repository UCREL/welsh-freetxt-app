import streamlit as st
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

WHITESPACE_HANDLER = lambda k: re.sub('\s+', ' ', re.sub('\n+', ' ', k.strip()))

def run_model(text):
    model_name = "csebuetnlp/mT5_multilingual_XLSum"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    input_ids = tokenizer([WHITESPACE_HANDLER(text)],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=512
    )["input_ids"]

    output_ids = model.generate(
        input_ids=input_ids,
        max_length=84,
        no_repeat_ngram_size=2,
        num_beams=4
    )[0]

    summary = tokenizer.decode(
        output_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    
st.title('Text Summarization Demo')
st.markdown('Using mT5 transformer model')
model = st.selectbox('Select the model', ("mT5"))

article_text = """Videos that say approved vaccines are dangerous and cause autism, cancer or infertility are among those that will be taken down, the company said.  The policy includes the termination of accounts of anti-vaccine influencers.  Tech giants have been criticised for not doing more to counter false health information on their sites.  In July, US President Joe Biden said social media platforms were largely responsible for people's scepticism in getting vaccinated by spreading misinformation, and appealed for them to address the issue.  YouTube, which is owned by Google, said 130,000 videos were removed from its platform since last year, when it implemented a ban on content spreading misinformation about Covid vaccines.  In a blog post, the company said it had seen false claims about Covid jabs "spill over into misinformation about vaccines in general". The new policy covers long-approved vaccines, such as those against measles or hepatitis B.  "We're expanding our medical misinformation policies on YouTube with new guidelines on currently administered vaccines that are approved and confirmed to be safe and effective by local health authorities and the WHO," the post said, referring to the World Health Organization."""

input_text = st.text_area('Text Input', article_text)



if st.button('Submit'):
    st.write(run_model(input_text))

#=======================================================================
# import torch
# from transformers import T5Tokenizer, T5ForConditionalGeneration

# if model == 'BART':
    # _num_beams = 4
    # _no_repeat_ngram_size = 3
    # _length_penalty = 1
    # _min_length = 12
    # _max_length = 128
    # _early_stopping = True
# else:
    # _num_beams = 4
    # _no_repeat_ngram_size = 3
    # _length_penalty = 2
    # _min_length = 30
    # _max_length = 200
    # _early_stopping = True

# col1, col2, col3 = st.beta_columns(3)
# _num_beams = col1.number_input("num_beams", value=_num_beams)
# _no_repeat_ngram_size = col2.number_input("no_repeat_ngram_size", value=_no_repeat_ngram_size)
# _length_penalty = col3.number_input("length_penalty", value=_length_penalty)

# col1, col2, col3 = st.beta_columns(3)
# _min_length = col1.number_input("min_length", value=_min_length)
# _max_length = col2.number_input("max_length", value=_max_length)
# _early_stopping = col3.number_input("early_stopping", value=_early_stopping)





# def run_model(input_text):
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # if model == "BART":
        # bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
        # bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
        # input_text = str(input_text)
        # input_text = ' '.join(input_text.split())
        # input_tokenized = bart_tokenizer.encode(input_text, return_tensors='pt').to(device)
        # summary_ids = bart_model.generate(input_tokenized,
                                          # num_beams=_num_beams,
                                          # no_repeat_ngram_size=_no_repeat_ngram_size,
                                          # length_penalty=_length_penalty,
                                          # min_length=_min_length,
                                          # max_length=_max_length,
                                          # early_stopping=_early_stopping)

        # output = [bart_tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces = False) for g in summary_ids]

        # st.write('Summary')
        # st.success(output[0])

    # else:
        # t5_model = T5ForConditionalGeneration.from_pretrained("t5-base")
        # t5_tokenizer = T5Tokenizer.from_pretrained("t5-base")
        # input_text = str(input_text).replace('\n', '')
        # input_text = ' '.join(input_text.split())
        # input_tokenized = t5_tokenizer.encode(input_text, return_tensors="pt").to(device)
        # summary_task = torch.tensor([[21603, 10]]).to(device)
        # input_tokenized = torch.cat([summary_task, input_tokenized], dim=-1).to(device)
        # summary_ids = t5_model.generate(input_tokenized,
                                        # num_beams=_num_beams,
                                        # no_repeat_ngram_size=_no_repeat_ngram_size,
                                        # length_penalty=_length_penalty,
                                        # min_length=_min_length,
                                        # max_length=_max_length,
                                        # early_stopping=_early_stopping)
        # output = [t5_tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]
        # st.write('Summary')
        # st.success(output[0])
