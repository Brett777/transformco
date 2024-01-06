import requests
import pandas as pd
from datetime import datetime, timedelta
import os
import datarobotx as drx
import pytz
import streamlit as st
import snowflake.connector
import re
st.set_page_config(page_title="TransformCo", layout="wide")

def mainPage():
    st.title("TransformCo")
    st.subheader("Ask a question about the business.")


# Main app
def _main():
    hide_streamlit_style = """
    <style>
    # MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)  # This let's you hide the Streamlit branding
    mainPage()


if __name__ == "__main__":
    _main()
