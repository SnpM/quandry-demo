import sys
import os

import streamlit as st
import pandas as pd

from quandry.classes import *
from quandry.evaluators import LlmClassifier_Gemini
from quandry.subjects import OpenAiSubject
from typing import *

import mock

default_prompt_df = [
    (f"{country} Capital", f"Ask for capital of {country}",
     f"What is the capital of {country}?", f"Answer correclty with '{capital}'") 
    for country,capital in mock.country_capitals.items()]
default_prompt_df = pd.DataFrame(default_prompt_df, columns=["Name", "Desc", "Prompt", "Expectation"])

def df2cases(df:pd.DataFrame) -> Collection[ExpectationCase]:
    return df.apply(lambda x: ExpectationCase(name=x.Name, desc=x.Desc, prompt=x.Prompt, expect=x.Expectation), axis=1)

def results2df(results:Collection[CaseResult]) -> pd.DataFrame:
    dicts = [
        x.__dict__
        for x in results
    ]
    return pd.DataFrame(dicts)


def main():
    st.set_page_config("Quandry Demo")
    st.title("Quandry Demo")

    #=====Prompt Package====
    st.subheader("Configure Prompts")
    # Streamlit hack: We need to store an original dataframe to make data_editor work correctly
    # Then save the edited output into our output session variable
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        st.session_state["og_cases_df"] = pd.read_csv(uploaded_file)
    elif "cases_df" not in st.session_state:
        st.session_state["og_cases_df"] = default_prompt_df.copy().reset_index(drop=True)
        
    og_cases_df = st.session_state["og_cases_df"]

    cases_df = st.data_editor(og_cases_df,num_rows="dynamic")
    
    st.session_state["cases_df"] = cases_df.copy()
    st.divider()

    #====Select Target====
    st.subheader("Configure Subject")
    subject_options = [mock.CapitalTriviaSubject, mock.HumanSubject, OpenAiSubject]

    if "subject_idx" not in st.session_state:
        st.session_state["subject_idx"] = 0
    subject_idx = st.session_state["subject_idx"]

    subject_idx = st.selectbox(
        "Select Subject", 
        range(len(subject_options)), 
        index=subject_idx, 
        format_func=lambda i:subject_options[i].__name__)
    st.session_state["subject_idx"] = subject_idx
    subject = subject_options[subject_idx]()

    if isinstance(subject, mock.HumanSubject):
        cases_df = st.session_state["cases_df"]

        # Storing original data_editor df in _ session variable
        if "_human_responses" not in st.session_state:
            st.session_state["_human_responses"] = pd.DataFrame(
                columns=["Prompt","Response"]).reset_index(drop=True)
        
        _human_responses:pd.DataFrame = st.session_state["_human_responses"]

        prompts = cases_df["Prompt"].values
        for prompt in prompts:
            if prompt is None:
                continue
            if prompt not in _human_responses["Prompt"]:
                _human_responses = pd.concat((
                    _human_responses,
                    pd.DataFrame({"Prompt":[prompt],"Response":[""]})
                    ))

    
        badprompts = _human_responses.apply(lambda row: row["Prompt"] not in prompts, axis=1)
        _human_responses = _human_responses[~badprompts]
    
        colconfig = {
            "Prompt":st.column_config.Column(disabled=True),
            "Response":st.column_config.Column(width="large")
        }
        human_responses = st.data_editor(
            _human_responses,num_rows="fixed", use_container_width=True,
            column_config=colconfig,
            hide_index=True,
        )
        st.session_state["human_responses"] = human_responses
        if len(human_responses) > 0:
            human_responses_dict = dict(zip(human_responses["Prompt"],human_responses["Response"]))
        else:
            human_responses_dict = {}
        subject.update(human_responses_dict)

        # TODO: Should we copy values to _human_responses df?
        # Currently _human_responses is not updating, but it needs to be the safe reference
        # st.dataframe(st.session_state["_human_responses"])
        # st.dataframe(st.session_state["human_responses"])

    st.divider()
    #====Select Evaluator====
    st.subheader("Configure Evaluator")
    evaluator_options = [mock.CapitalTriviaEvaluator, LlmClassifier_Gemini]

    if "evaluator_idx" not in st.session_state:
        st.session_state["evaluator_idx"] = 0
    evaluator_idx = st.session_state["evaluator_idx"]

    evaluator_idx = st.selectbox(
        "Select Evaluator", 
        range(len(evaluator_options)), 
        index=evaluator_idx, 
        format_func=lambda i:evaluator_options[i].__name__)
    
    st.session_state["evaluator_idx"] = evaluator_idx
    evaluator = evaluator_options[evaluator_idx]()

    st.divider()
    #====Generate Report====
    st.subheader("Report")
    if st.button("Evaluate"):
        # Construct subject and evaluator with no parameters
        # TODO: Enable passing a kwargs in; maybe define schema
        tester = ExpectationTester(subject, evaluator)
        cases = df2cases(st.session_state["cases_df"])
        results = tester.test_batch(cases)
        st.session_state["results_df"] = results2df(results)
    
    if "results_df" in st.session_state:
        results_df = st.session_state["results_df"]
        results_df["evalcode_name"] = results_df["evalcode"].apply(lambda x: EvalCode(x).name)
        st.dataframe(results_df[['prompt','response','evalcode_name','explanation']])
    
    
if __name__ == "__main__":
    main()
