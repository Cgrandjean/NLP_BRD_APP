import streamlit as st
import numpy as np
import pandas as pd
import os

st.set_page_config(
    page_title="High level Sentiment analysis",
)

df_negative=pd.read_csv(os.path.join('./data',"Negative","Negative"+".csv"))
df_neutral=pd.read_csv(os.path.join('./data',"Neutral","Neutral"+".csv"))
df_positive=pd.read_csv(os.path.join('./data',"Positive","Positive"+".csv"))
df_mixed=pd.read_csv(os.path.join('./data',"Mixed","Mixed"+".csv"))


st.markdown("# Sentiment perception of the app")

df=pd.DataFrame([
    [int(len(df_negative)), int(len(df_neutral)),int(len(df_positive)), int(len(df_mixed))],
    [len(df_negative)/len(df_mixed), len(df_neutral)/len(df_mixed),len(df_positive)/len(df_mixed), len(df_mixed)/len(df_mixed)]
    ],
    columns=["Negative","Neutral","Positive","Total"],
    index=["Samples","Percentage(%)"]
)
st.dataframe(df)

st.markdown("""Overall the application is clearly **poorly** perceived.
            Here are few samples from each sentiment clusters:
            """)
            
st.markdown("""#### - Negative samples:""")            
st.dataframe(df_negative.sample(5).loc[:,["Date","App","Translated_Content","Content_filtered","Topics_Names"]])
st.markdown("""#### - Neutral samples:""")            
st.dataframe(df_neutral.sample(5).loc[:,["Date","App","Translated_Content","Content_filtered","Topics_Names"]])
st.markdown("""#### - Positive samples:""")           
st.dataframe(df_positive.sample(5).loc[:,["Date","App","Translated_Content","Content_filtered","Topics_Names"]]) 