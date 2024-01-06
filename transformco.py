import requests
import pandas as pd
from datetime import datetime, timedelta
import os
import datarobotx as drx
import streamlit as st
from snowflake.sqlalchemy import URL
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError

st.set_page_config(page_title="TransformCo", layout="wide")


def getSnowflakeSQL(prompt):
    '''
    Submits the user's prompt to DataRobot, gets Snowflake SQL
    '''
    data = pd.DataFrame({"promptText": [prompt]})
    API_URL = 'https://cfds-ccm-prod.orm.datarobot.com/predApi/v1.0/deployments/{deployment_id}/predictions'
    API_KEY = os.environ["DATAROBOT_API_TOKEN"]
    DATAROBOT_KEY = os.environ["DATAROBOT_KEY"]
    deployment_id = '6596fe9774e6d5a9236e00a9'
    headers = {
        'Content-Type': 'application/json; charset=UTF-8',
        'Authorization': 'Bearer {}'.format(API_KEY),
        'DataRobot-Key': DATAROBOT_KEY,
    }
    url = API_URL.format(deployment_id=deployment_id)
    predictions_response = requests.post(
        url,
        data=data.to_json(orient='records'),
        headers=headers
    )
    return predictions_response.json()["data"][0]["prediction"]

def executeSnowflakeQuery(snowflakeSQL):
    '''
    Executes the Snowflake SQL generated by the LLM
    '''
    user = os.environ["snowflakeUser"]
    password = os.environ["snowflakePassword"]
    account = os.environ["snowflakeAccount"]
    warehouse = os.environ["snowflakeWarehouse"]
    database = os.environ["snowflakeDatabase"]
    schema = 'TRANSFORM'

    # Create a connection
    engine = create_engine(URL(
        user=user,
        password=password,
        account=account,
        warehouse=warehouse,
        database=database,
        schema=schema
    ))
    results = None
    try:
        with engine.connect() as connection:
           results = pd.read_sql_query(snowflakeSQL, connection)
    except SQLAlchemyError as e:
        print(f"An error occurred: {e}")
    finally:
        engine.dispose()
    return results


def getBusinessAnalysis(prompt):
    '''
    Given the question, the Snowflake SQL, and the response, retrieve the business analysis and suggestions.
    '''
    data = pd.DataFrame({"promptText": [prompt]})
    API_URL = 'https://cfds-ccm-prod.orm.datarobot.com/predApi/v1.0/deployments/{deployment_id}/predictions'
    API_KEY = os.environ["DATAROBOT_API_TOKEN"]
    DATAROBOT_KEY = os.environ["DATAROBOT_KEY"]
    deployment_id = '659842b929da7d595d6df6b0'
    headers = {
        'Content-Type': 'application/json; charset=UTF-8',
        'Authorization': 'Bearer {}'.format(API_KEY),
        'DataRobot-Key': DATAROBOT_KEY,
    }
    url = API_URL.format(deployment_id=deployment_id)
    predictions_response = requests.post(
        url,
        data=data.to_json(orient='records'),
        headers=headers
    )

    return predictions_response.json()["data"][0]["prediction"]


def mainPage():
    st.title("TransformCo")
    st.subheader("Ask a question about the business.")
    prompt = st.text_input(label="Question")
    submitQuestion = st.button(label="Ask")

    if submitQuestion:
        with st.spinner(text="Generating Query..."):
            with st.expander(label="Snowflake SQL", expanded=True):
                snowflakeSQL = getSnowflakeSQL(prompt)
                st.code(body=snowflakeSQL, language="sql")
        with st.spinner(text="Executing Query..."):
            with st.expander(label="Query Result", expanded=True):
                answer = executeSnowflakeQuery(snowflakeSQL)
                st.write(answer)
        with st.spinner(text="Analyzing..."):
            with st.expander(label="Analysis", expanded=True):
                analysis = getBusinessAnalysis(prompt + str(snowflakeSQL) + str(answer))
                st.markdown(analysis)






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
