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

def getChartCode(prompt):
    '''
        Given the question, the Snowflake SQL, and the response, retrieve the chart code.
    '''
    data = pd.DataFrame({"promptText": [prompt]})
    API_URL = 'https://cfds-ccm-prod.orm.datarobot.com/predApi/v1.0/deployments/{deployment_id}/predictions'
    API_KEY = os.environ["DATAROBOT_API_TOKEN"]
    DATAROBOT_KEY = os.environ["DATAROBOT_KEY"]
    deployment_id = '659f27d4c66e9cd86ce9133a'
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


def generateSQLandResult(prompt, attemptCount):

    if attemptCount == 1:
        spinnerText = ""
    if attemptCount == 2:
        spinnerText = "Working on an a better approach..."
    if attemptCount == 3:
        spinnerText = "Trying a different solution..."

    with st.spinner(text="Generating Query... " + spinnerText):
        snowflakeSQL = getSnowflakeSQL(prompt)

    with st.spinner(text="Executing Query..."):
        answer = executeSnowflakeQuery(snowflakeSQL)
    return snowflakeSQL, answer

def mainPage():
    st.title("TransformCo")
    st.subheader("Ask a question about the business.")
    prompt = st.text_input(label="Question")
    submitQuestion = st.button(label="Ask")

    if submitQuestion:
        attemptCount = 1
        snowflakeSQL, answer =  generateSQLandResult(prompt, attemptCount)
        if answer is None or answer.empty:
            attemptCount = 2
            snowflakeSQL, answer = generateSQLandResult(prompt, attemptCount)
        if answer is None or answer.empty:
            attemptCount = 3
            snowflakeSQL, answer = generateSQLandResult(prompt, attemptCount)

        with st.expander(label="Snowflake SQL", expanded=True):
            st.code(body=snowflakeSQL, language="sql")

        with st.expander(label="Query Result", expanded=True):
            if attemptCount <= 3:
                st.dataframe(answer.reset_index(drop=True))
            else: st.write("Query produced no result")

        with st.spinner(text="Visualizing..."):
            with st.expander(label="Visualization", expanded=True):
                import plotly.graph_objects as go

                chartCode = getChartCode(prompt + str(snowflakeSQL) + str(answer))
                st.text(chartCode.replace("```python","").replace("```",""))
                chartCode = chartCode.replace("```python","").replace("```","")
                function_dict = {}
                exec(chartCode, function_dict) # execute the code created by our LLM
                create_charts = function_dict['create_charts'] # get the function that our code created
                fig1, fig2 = create_charts(answer)
                st.plotly_chart(fig1, use_container_width=True)
                st.plotly_chart(fig2, use_container_width=True)


        with st.spinner(text="Analyzing..."):
            with st.expander(label="Analysis", expanded=True):
                analysis = getBusinessAnalysis(prompt + str(snowflakeSQL) + str(answer))
                st.markdown(analysis.replace("$","\$"))






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
