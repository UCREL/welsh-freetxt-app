import os
import numpy as np
import streamlit as st
import nltk
import networkx as nx
nltk.download('punkt') # one time execution
from nltk.tokenize import sent_tokenize
from lexrank import LexRank
from summa.summarizer import summarize as summa_summarizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

#=========================
example_text = """Mae Erthygl 25 o Ddatganiad Cyffredinol Hawliau Dynol 1948 y Cenhedloedd Unedig yn nodi: "Mae gan bawb yr hawl i safon byw sy'n ddigonol ar gyfer iechyd a lles ei hun a'i deulu, gan gynnwys bwyd, dillad, tai a gofal meddygol a gwasanaethau cymdeithasol angenrheidiol.
"Mae'r Datganiad Cyffredinol yn cynnwys lletyaeth er mwyn diogelu person ac mae hefyd yn sôn yn arbennig am y gofal a roddir i'r rheini sydd mewn mamolaeth neu blentyndod.
Ystyrir mai Datganiad Cyffredinol o Hawliau Dynol fel y datganiad rhyngwladol cyntaf o hawliau dynol sylfaenol.
Dywedodd Uchel Gomisiynydd y Cenhedloedd Unedig dros Hawliau Dynol Navanethem Pillay fod y Datganiad Cyffredinol o Hawliau Dynol.
"yn ymgorffori gweledigaeth sy'n gofyn am gymryd yr holl hawliau dynol - sifil, gwleidyddol, economaidd, cymdeithasol neu ddiwylliannol - fel cyfanwaith anwahanadwy ac organig, anwahanadwy a rhyngddibynnol."""
example_summary = """Mae Datganiad Cyffredinol Hawliau Dynol 1948 yn dweud bod gan bawb yr hawl i safon byw digonol.
Mae hynny yn cynnwys mynediad at fwyd a dillad a gofal meddygol i bob unigolyn. Dyma’r datganiad cyntaf o hawliau dynol"""

## Define summarizer models

# lex_rank
def lex_rank_summarize(article, ratio=0.5):
  sentences = sent_tokenize(article)
  summary = LexRank(sentences).get_summary(sentences,
                             summary_size=int(len(sentences)*0.5), threshold=.1)
  return "\n".join(summary)

# text_rank
def text_rank_summarize(article, ratio):
  return summa_summarizer(article, ratio=ratio)

# text_rank
# def text_rank_summarize(article, ratio=0.5):
  # return summa_summarizer(article, ratio=ratio)

# Define Topline summarizers
def tfidf_summarize(article, ratio=0.5):
  sentences = sent_tokenize(article)
  # get similarity matrix
  sim_mat = cosine_similarity(TfidfVectorizer().fit_transform(sentences))
  scores = nx.pagerank_numpy(nx.from_numpy_array(sim_mat))
  top_ranked = sorted(scores.items(), key=lambda x: x[1], 
                      reverse=True)[:int(len(scores)*ratio)]
  summary = [sentences[i] for i,_ in top_ranked]
  return "\n".join(summary)

# build similarity matrix
def gen_similarity_matrix(sents):
  sim_mat = np.zeros([len(sents), len(sents)])
  for i in range(len(sents)):
    for j in range(len(sents)):
      if i != j:
        sim_mat[i][j] = float(cosine_similarity(sents[i], sents[j]))
  return sim_mat
