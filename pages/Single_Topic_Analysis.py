import streamlit as st
import numpy as np
import pandas as pd
import os 
from bertopic import BERTopic
import bertopic
from sentence_transformers import SentenceTransformer
from bertopic.backend._utils import select_backend


def load_data(add_selectbox):
    df_selected=pd.read_csv(os.path.join('./data',add_selectbox,add_selectbox+".csv"))
    path=os.path.join(r'./data',add_selectbox,add_selectbox+"_model")
    sentence_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
    model = select_backend(sentence_model)
    topic_model = BERTopic.load(path, embedding_model=model)
    return df_selected,topic_model 


@st.cache
def get_topics(docs):
    topics=topic_model.get_topic_info().set_index('Topic')
    return topics

def retrieve_whole_sentence_representative(topic_model,df,topic_num):
  representative_docs=topic_model.get_representative_docs(topic_num)
  contextual_l=[]
  for doc in representative_docs:
    line=df_selected.loc[df_selected.Content_filtered==doc,:]
    contextual_l.append([line.Date.iloc[0],
                         line.Translated_Content.iloc[0],\
                         doc,\
                         line.App.iloc[0],\
                         line.Topics.iloc[0],\
                         line.Topics_Names.iloc[0]])
  contextual_df=pd.DataFrame(contextual_l,columns=["Date","Context","Analyzed sentence","App","Topics","Topics_Names"])
  return contextual_df

def retrieve_topic_examples(df,topic):
  mask=df["Topics"]==topic
  df_examples=df.loc[mask,["Date","Translated_Content","Content_filtered","App","Topics","Topics_Names"]]
  df_examples.columns=["Date","Context","Analyzed sentence","App","Topics","Topics_Names"]
  return df_examples
st.set_page_config(
    page_title="Single topic analysis",
)

st.sidebar.markdown("### Sentiment selection")
add_selectbox=st.sidebar.radio(
    'What Part of data do you want to analyze?',
    ('Negative', 'Neutral', 'Positive','Mixed'))

df_selected,topic_model = load_data(add_selectbox)
classes=list(df_selected.Label)
docs=list(df_selected.Content_filtered)
timestamps=list(df_selected.Date)


df_topics=get_topics(docs)
st.markdown("## Single Topic analysis")
Lookup_terms=st.text_input(label='Terms to look up for ?', value="")
if Lookup_terms:
    df_selected_topics=df_topics.loc[topic_model.find_topics(Lookup_terms)[0],:].sort_values('Count',ascending=False)
    st.dataframe(df_selected_topics)
    st.plotly_chart(topic_model.visualize_barchart(df_selected_topics.index))
    
    
    st.markdown("## Topic in detail")
    Lookup_topic=st.text_input(label='Number of the topic you want to investigate', value="")
    if Lookup_topic.isnumeric():
        st.markdown("### Representative Documents")
        contextual_df=retrieve_whole_sentence_representative(topic_model,df_selected,int(Lookup_topic))
        st.dataframe(contextual_df)
        st.markdown("### Documents in the topic")
        topic_df=retrieve_topic_examples(df_selected,int(Lookup_topic))
        st.dataframe(topic_df)
        
    
    