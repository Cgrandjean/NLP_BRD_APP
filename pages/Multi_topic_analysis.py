import streamlit as st
from bertopic import BERTopic
import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from bertopic.backend._utils import select_backend
st.set_page_config(
    page_title="Multi-topic analysis",
)

# @st.cache
def load_data(add_selectbox):
    df_selected=pd.read_csv(os.path.join('./data',add_selectbox,add_selectbox+".csv"))
    path=os.path.join(r'./data',add_selectbox,add_selectbox+"_model")
    sentence_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
    model = select_backend(sentence_model)
    topic_model = BERTopic.load(path, embedding_model=model)
    return df_selected,topic_model 

# @st.cache
def get_topics(nr_topics,docs):
    topic_model.reduce_topics(docs, nr_topics=nr_topics)
    topics=topic_model.get_topic_info().set_index('Topic')
    return topics,topic_model

# @st.cache
def get_topics_over_time(topic_model,docs,timestamps):
    topics_over_time=topic_model.topics_over_time(docs,timestamps,nr_bins=50)
    return topics_over_time


def get_topics_hierarchized(topic_model,docs):
    hierarchical_topics=topic_model.hierarchical_topics(docs)
    return topic_model.visualize_hierarchy(hierarchical_topics=hierarchical_topics)

st.sidebar.markdown("### Sentiment selection")

add_selectbox=st.sidebar.radio(
    'What Part of data do you want to analyze?',
    ('Negative', 'Neutral', 'Positive','Mixed'))

df_selected,topic_model = load_data(add_selectbox)
classes=list(df_selected.Label)
docs=list(df_selected.Content_filtered)
timestamps=list(df_selected.Date)


st.markdown("# Multi Topic analysis")
nr_topics = st.slider('Reduce maximum number of topics',min_value=1,max_value=200, value=20)

st.markdown("### Topics Found")
df_topics,topic_model_reduced=get_topics(nr_topics,docs)
st.dataframe(df_topics)
st.plotly_chart(topic_model_reduced.visualize_topics())
    
st.markdown("### Topics hierarchized")
st.plotly_chart(get_topics_hierarchized(topic_model_reduced,docs))

st.markdown("# In depth topics")
num_topics=st.text_input(label='How many topics do you want to detail ?', value="")
if num_topics.isnumeric():
    st.markdown("### Topics detailed")
    st.plotly_chart(topic_model_reduced.visualize_barchart(top_n_topics = int(num_topics)))
    st.markdown("### Topics over time")
    topics_over_time=get_topics_over_time(topic_model_reduced,docs,timestamps)
    st.plotly_chart(topic_model_reduced.visualize_topics_over_time(topics_over_time\
                                                           ,top_n_topics=int(num_topics)))
if add_selectbox=='Mixed':
    st.markdown("### Topics by class sentiment")
    topics_per_class=topic_model_reduced.topics_per_class(docs,classes=classes)
    st.plotly_chart(topic_model_reduced.visualize_topics_per_class(topics_per_class,top_n_topics=int(num_topics)))
    
