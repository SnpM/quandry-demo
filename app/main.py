import sys
import os

import streamlit as st
import pandas as pd
from streamlit.runtime.scriptrunner import add_script_run_ctx
from threading import Thread

from quandry.classes import *
from quandry.evaluators import LlmClassifier_Gemini, LlmClassifier_ChatGPT
from quandry.subjects import OpenAiSubject, GeminiSubject
from typing import *
import time

import mock
import streamlit.components.v1 as components
from streamlit_scroll_navigation import scroll_navbar


class SubjectOption:
    def __init__(self, name:str, subject:ISubject):
        self.name = name
        self.subject = subject
subject_options = [
    SubjectOption("gpt-4o-mini", OpenAiSubject("gpt-4o-mini-2024-07-18")),
    SubjectOption("gpt-4o", OpenAiSubject("gpt-4o-2024-05-13")),
    SubjectOption("gemini-1.5-flash", GeminiSubject("gemini-1.5-flash")),
    SubjectOption("gpt-3.5-turbo-0125", OpenAiSubject("gpt-3.5-turbo-0125")),
    SubjectOption("gpt-3.5-turbo-1106", OpenAiSubject("gpt-3.5-turbo-1106")),
    SubjectOption("gpt-4", OpenAiSubject("gpt-4-0613")),
    SubjectOption("gpt-4-turbo", OpenAiSubject("gpt-4-turbo-2024-04-09")),
    SubjectOption("Human", mock.HumanSubject()),
    #SubjectOption("Capital Trivia", mock.CapitalTriviaSubject())           
] 

class EvaluatorOption:
    def __init__(self, name:str, evaluator:IEvaluator):
        self.name = name
        self.evaluator = evaluator
evaluator_options = [
    EvaluatorOption("Llm Classifier (gpt-4o-mini)",LlmClassifier_ChatGPT(model_id="gpt-4o-mini")),
    EvaluatorOption("Llm Classifier (gpt-4o)",LlmClassifier_ChatGPT()),
    EvaluatorOption("Llm Classifier (gemini-1.5-flash)",LlmClassifier_Gemini()),
    #mock.CapitalTriviaEvaluator
    ]


default_prompt_df = [
    (f"{country} Capital", f"Ask for capital of {country}",
     f"What is the capital of {country}?", f"Provide the correct information, '{capital}'") 
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
    st.set_page_config("Quandry Demo", layout="wide", initial_sidebar_state="expanded")
    with st.sidebar:
        version = "demo v0.2.0"
        st.markdown("<h1 style='text-align: center; font-size: 3em;'>Quandry</h1>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='text-align: center;'>{version}</h3>", unsafe_allow_html=True)
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
        prompt_selection = st.selectbox("Select a default prompt package", ["Jailbreaks","Comprehensive","Country Capitals"])
        if prompt_selection == "Jailbreaks":
            st.session_state["cases_df_anchor"] = pd.read_csv("prompts/jailbreaks.csv")
        if prompt_selection == "Comprehensive":
            st.session_state["cases_df_anchor"] = pd.read_csv("prompts/comprehensive.csv")
        elif prompt_selection == "Country Capitals":
            st.session_state["cases_df_anchor"] = default_prompt_df.copy().reset_index(drop=True)
        
    cases_df_anchor:pd.DataFrame = st.session_state["cases_df_anchor"]

    with st.expander("Edit Prompts", expanded=True):
        colconfig = {
            "Desc":st.column_config.Column(width="small"),
            "Prompt":st.column_config.Column(width="small"),
            "Expectation":st.column_config.Column(width="large")
        }
        cases_df = st.data_editor(
            cases_df_anchor,num_rows="dynamic", use_container_width=True,
            column_config=colconfig,
        )
        st.session_state["cases_df"] = cases_df
        
    st.divider()

    #====Select Target====
    st.subheader("Configure Subject", anchor="Configure-Subject")


    st.session_state["subject_idx"] = st.session_state.get("subject_idx", 0)
    subject_idx = st.session_state["subject_idx"]

    subject_idx = st.selectbox(
        "Select Subject", 
        range(len(subject_options)),
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
            key="human_responses_data_editor"
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

    if "evaluator_idx" not in st.session_state:
        st.session_state["evaluator_idx"] = 0
    evaluator_idx = st.session_state["evaluator_idx"]

    evaluator_idx = st.selectbox(
        "Select Evaluator", 
        range(len(evaluator_options)),
        format_func=lambda i:evaluator_options[i].name)
    
    st.session_state["evaluator_idx"] = evaluator_idx
    evaluator = evaluator_options[evaluator_idx].evaluator

    st.divider()
    #====Generate Report====
    if "reports" not in st.session_state:
        st.session_state["reports"] = []
        
    st.subheader("Generate Report", anchor="Generate-Report")
    
    # Set default current_run
    st.session_state["current_run"] = st.session_state.get("current_run", None)
    st.session_state["current_run_thread"] = st.session_state.get("current_run_thread", None)
    st.session_state["start_run"] = st.session_state.get("start_run", False)
        
    def check_run() -> bool:
        if st.session_state["start_run"]:
            print ("Starting run")
            st.session_state["start_run"] = False
            subject_idx = st.session_state["subject_idx"]
            subject_name = subject_options[subject_idx].name
            subject = subject_options[subject_idx].subject
            evaluator_idx = st.session_state["evaluator_idx"]
            evaluator = evaluator_options[evaluator_idx].evaluator
            
            def run(subject_name, subject):
                tester = ExpectationTester(subject, evaluator)
                cases = df2cases(st.session_state["cases_df"])
                try:
                    results = tester.test_batch(cases)
                    results_df = results2df(results)
            
                    # Format results df
                    results_df["evalcode_name"] = results_df["evalcode"].apply(lambda x: EvalCode(x).name)
                    
                    # Add to session state results
                    print ("Adding to results dict in session state")
                    # Add to beginning of reports
                    report_info = (subject_name,results_df)
                    st.session_state["reports"].insert(0, report_info)
                    st.session_state["force_tab"] = subject_name
                except Exception as e:
                    print(f"Error running evaluation: {e}")
                    st.session_state["run_error"] = str(e)
                    import traceback
                    traceback.print_exc()
                finally:
                    st.session_state["current_run"] = None
                    st.session_state["current_run_thread"] = None   
                    print("run thread done")
                        
            # Check if already running
            if st.session_state["current_run"] is None:
                st.session_state["current_run"] = subject_name
                t = Thread(target=run, args=(subject_name, subject))
                st.session_state["current_run_thread"] = t
                add_script_run_ctx(t)
                t.start()
            else:
                print("Already running")
            return True
        else:
            return st.session_state["current_run"] is not None
    button_container = st.empty()
    def draw_button():
        with button_container:
            button_disabled = st.session_state["current_run"] is not None or st.session_state["start_run"]
            if st.button("Run Evaluation", disabled=button_disabled, key="run_button"):
                st.session_state["start_run"] = True    
                print("click")
                st.rerun()
    
    status_container = st.empty()
    
    # When force_tab is set, force the first tab to be selected
    force_tab = st.session_state.get("force_tab", None)
    js_code = "<script>frameElement.parentElement.style.display = 'none';</script>"
    if force_tab is not None:
        st.session_state["force_tab"] = None
        # Little hack to reset selection to first tab
        js_code = """
            <script>
                parentDiv=frameElement.parentElement
                parentDiv.nextElementSibling.querySelector('[data-testid="stTab"]').click()
            </script>
        """
    components.html(js_code)
    
    reports_container = st.empty()
    def draw_report():
        
        with reports_container:
            # Display all reports in one tab per report  
            reports = st.session_state["reports"]
            tab_labels = [x[0] for x in reports]
            
            if len(tab_labels) > 0:
                tabs = st.tabs(tab_labels)
                i = 0
                for report_info, tab in zip(reports, tabs):
                    with tab:
                        result_df = report_info[1]
                        column_config={
                            "evalcode_name":st.column_config.Column(width="small"),
                            "prompt":st.column_config.Column(width="small"),
                            "response":st.column_config.Column(width="small"),
                            "explanation":st.column_config.Column(width="large")
                        }
                        def highlight_evalcode(val):
                            color = ''
                            if val == 'PASS':
                                color = 'background-color: green'
                            elif val == 'FAIL':
                                color = 'background-color: red'
                            return color

                        styled_df = result_df[['name','prompt', 'response', 'evalcode_name', 'explanation']].style.map(highlight_evalcode, subset=['evalcode_name'])

                        st.dataframe(
                            styled_df, 
                            use_container_width=True, 
                            hide_index=True,
                            column_config=column_config,
                            key = i)
                        i += 1
    draw_button()
    running = check_run()
    
    if running:
        draw_report()
        current_run = st.session_state["current_run"]
        t = st.session_state["current_run_thread"]
        with status_container:
            with st.spinner(f"Running evaluation on {current_run}..."): 
                while t and t.is_alive():
                    time.sleep(.25)
                print("spin done for ", t)
        st.rerun()
    else:
        draw_report()
        if "run_error" in st.session_state:
            with status_container:
                st.markdown(f"<span style='color:red;'>Error running evaluation: {st.session_state['run_error']}</span>", unsafe_allow_html=True)
            del st.session_state["run_error"]
        else:
            status_container.text("")
    
if __name__ == "__main__":
    main()
