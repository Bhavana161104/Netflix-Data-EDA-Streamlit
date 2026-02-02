import streamlit as st
from eda import (
    load_data,
    plot_content_type,
    plot_top_genres,
    plot_release_year
)
from model import run_model

st.set_page_config(page_title="Netflix EDA", layout="wide")

st.title("📺 Netflix Data Analysis Dashboard")

# Load data
df = load_data()

st.subheader("Dataset Preview")
st.dataframe(df.head())

col1, col2 = st.columns(2)

with col1:
    st.pyplot(plot_content_type(df))

with col2:
    st.pyplot(plot_top_genres(df))

st.pyplot(plot_release_year(df))

st.subheader("Machine Learning Result")
accuracy = run_model(df)
st.success(f"Model Accuracy: {accuracy:.2f}")

