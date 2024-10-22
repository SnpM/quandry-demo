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
    else:
        prompt_selection = st.selectbox("Select a default prompt package", ["Jailbreaks","Country Capitals"])
        if prompt_selection == "Jailbreaks":
            st.session_state["cases_df_anchor"] = pd.read_csv("prompts/jailbreaks.csv")
        elif prompt_selection == "Country Capitals":
            st.session_state["cases_df_anchor"] = default_prompt_df.copy().reset_index(drop=True)
        
    cases_df_anchor:pd.DataFrame = st.session_state["cases_df_anchor"]

    # Hide by default
    with st.expander("Edit Prompts", expanded=True):
        cases_df = st.data_editor(cases_df_anchor,num_rows="dynamic", use_container_width=True)
        st.session_state["cases_df"] = cases_df
        
    st.divider()

    #====Select Target====
    st.subheader("Configure Subject", anchor="Configure-Subject")
    class SubjectOption:
        def __init__(self, name:str, subject:ISubject):
            self.name = name
            self.subject = subject
        
    subject_options = [
        SubjectOption("gpt-4o-mini", OpenAiSubject("gpt-4o-mini-2024-07-18")),   
        SubjectOption("gpt-3.5-turbo-0125", OpenAiSubject("gpt-3.5-turbo-0125")),
        SubjectOption("o1-mini", OpenAiSubject("o1-mini-2024-09-12")),
        SubjectOption("Human", mock.HumanSubject()),
        SubjectOption("Capital Trivia", mock.CapitalTriviaSubject())           
    ] 

    if "subject_idx" not in st.session_state:
        st.session_state["subject_idx"] = 0
    subject_idx = st.session_state["subject_idx"]

    subject_idx = st.selectbox(
        "Select Subject", 
        range(len(subject_options)), 
        index=subject_idx, 
        format_func=lambda i:subject_options[i].name)
    st.session_state["subject_idx"] = subject_idx
    subject = subject_options[subject_idx].subject

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
    if "reports" not in st.session_state:
        st.session_state["reports"] = []
        
    force_tab = None
    st.subheader("Generate Report", anchor="Generate-Report")
    if st.button("Run Evaluation"):
        # Construct subject and evaluator with no parameters
        # TODO: Enable passing a kwargs in; maybe define schema
        subject_name = subject_options[subject_idx].name
        subject = subject_options[subject_idx].subject
        tester = ExpectationTester(subject, evaluator)
        cases = df2cases(st.session_state["cases_df"])
        results = tester.test_batch(cases)
        results_df = results2df(results)
    
        # Format results df
        results_df["evalcode_name"] = results_df["evalcode"].apply(lambda x: EvalCode(x).name)
        
        # Add to session state results
        print ("Adding to results dict in session state")
        # Add to beginning of reports
        report_info = (subject_name,results_df)
        st.session_state["reports"].insert(0, report_info)
        force_tab = subject_name
        
    # Display all reports in one tab per report
    
    
    reports = st.session_state["reports"]
    tab_labels = [x[0] for x in reports]
    
    if len(tab_labels) > 0:
        if force_tab is not None:
            # Little hack to select first anchor
            components.html("<script>frameElement.parentElement.style.display = 'none';</script>",)
            
        tabs = st.tabs(tab_labels)
        for report_info, tab in zip(reports, tabs):
            with tab:
                result_df = report_info[1]
                column_config={
                    "explanation":st.column_config.Column(width="large"),
                    "evalcode_name":st.column_config.Column(width="small")
                }
                st.dataframe(
                    result_df[['prompt', 'response', 'evalcode_name', 'explanation']], 
                    use_container_width=True, 
                    hide_index=True,
                    column_config=column_config
                )
            
    
if __name__ == "__main__":
    main()
