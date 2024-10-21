import sys
import os

import streamlit as st
import pandas as pd

from quandry.classes import *
from quandry.evaluators import LlmClassifier_Gemini
from quandry.subjects import OpenAiSubject
from typing import *

import mock
import streamlit.components.v1 as components
from streamlit_scroll_navigation import scroll_navbar

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
        
nav_labels = ["Configure Prompts", "Configure Subject", "Configure Evaluator", "Generate Report"]
nav_anchors = ["Configure-Prompts", "Configure-Subject", "Configure-Evaluator", "Generate-Report"]
def main():
    st.set_page_config("Quandry Demo", layout="wide")

    with st.sidebar:
        st.markdown("<h1 style='text-align: center; font-size: 3em;'>Quandry</h1>", unsafe_allow_html=True)
        st.markdown("<h3 style='text-align: center;'>demo v0.1.0</h3>", unsafe_allow_html=True)
        st.divider()
        scroll_navbar(nav_anchors, anchor_labels=nav_labels)
        
    #=====Prompt Package====
    st.subheader("Configure Prompts",anchor="Configure-Prompts")

    uploaded_file = st.file_uploader(
        "Choose a CSV file", type="csv",
        accept_multiple_files=False,
    )
    
    if uploaded_file is not None:
        st.session_state["cases_df_anchor"] = pd.read_csv(uploaded_file)
    elif "cases_df" not in st.session_state:
        st.session_state["cases_df_anchor"] = default_prompt_df.copy().reset_index(drop=True)
        
    cases_df_anchor:pd.DataFrame = st.session_state["cases_df_anchor"]

    # Hide by default
    with st.expander("Edit Prompts", expanded=False):
        cases_df = st.data_editor(cases_df_anchor,num_rows="dynamic", use_container_width=True)
        st.session_state["cases_df"] = cases_df
        
    st.divider()

    #====Select Target====
    st.subheader("Configure Subject", anchor="Configure-Subject")
    subject_options = [OpenAiSubject, mock.CapitalTriviaSubject, mock.HumanSubject]

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
        if "human_responses_anchor" not in st.session_state:
            st.session_state["human_responses_anchor"] = pd.DataFrame(
                columns=["Prompt","Response"]).reset_index(drop=True)
        
        human_responses_anchor:pd.DataFrame = st.session_state["human_responses_anchor"]
        def clean_response_df(prompts, responses):
            for prompt in prompts:
                if prompt is None:
                    continue
                if "Prompt" not in responses.columns:
                    import pdb; pdb.set_trace()
                    
                # Add row if prompt not in responses
                if prompt not in responses["Prompt"].values:
                    responses = pd.concat([responses, pd.DataFrame([{"Prompt": prompt, "Response": ""}])], ignore_index=True)
                    
            # Drop rows that are not in prompts
            responses = responses[responses["Prompt"].isin(prompts)]
            return responses
        human_responses_anchor = clean_response_df(cases_df["Prompt"].values, human_responses_anchor)
        st.session_state["human_responses_anchor"] = human_responses_anchor


        colconfig = {
            "Prompt":st.column_config.Column(disabled=True),
            "Response":st.column_config.Column(width="large")
        }
        human_responses = st.data_editor(
            human_responses_anchor,
            num_rows="fixed",
            use_container_width=True,
            column_config=colconfig,
            hide_index=True,
        )
        human_responses = clean_response_df(cases_df["Prompt"].values, human_responses)
        human_responses_anchor.update(human_responses)
        
        st.session_state["human_responses"] = human_responses
        human_responses_dict = {}
        for idx,row in human_responses.iterrows():
            human_responses_dict[row["Prompt"]] = row["Response"]
        subject.update(human_responses_dict)

        # TODO: Should we copy values to _human_responses df?
        # Currently _human_responses is not updating, but it needs to be the safe reference
        # st.dataframe(st.session_state["_human_responses"])
        # st.dataframe(st.session_state["human_responses"])

    st.divider()
    #====Select Evaluator====
    st.subheader("Configure Evaluator", anchor="Configure-Evaluator")
    evaluator_options = [LlmClassifier_Gemini, mock.CapitalTriviaEvaluator]

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
    st.subheader("Generate Report", anchor="Generate-Report")
    if st.button("Run Evaluation"):
        # Construct subject and evaluator with no parameters
        # TODO: Enable passing a kwargs in; maybe define schema
        tester = ExpectationTester(subject, evaluator)
        cases = df2cases(st.session_state["cases_df"])
        results = tester.test_batch(cases)
        st.session_state["results_df"] = results2df(results)
    
    if "results_df" in st.session_state:
        results_df = st.session_state["results_df"]
        results_df["evalcode_name"] = results_df["evalcode"].apply(lambda x: EvalCode(x).name)
        column_config={
            "explanation":st.column_config.Column(width="large"),
            "evalcode_name":st.column_config.Column(width="small")
        }
        st.dataframe(
            results_df[['prompt', 'response', 'evalcode_name', 'explanation']], 
            use_container_width=True, 
            hide_index=True,
            column_config=column_config
        )
            
    
if __name__ == "__main__":
    main()
