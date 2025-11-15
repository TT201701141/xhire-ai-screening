import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import random
from xhire_utils import (
    extract_text, calculate_score, extract_skills, 
    explain_score_lime, detect_bias, DEFAULT_JOB_PROFILE
)

# --- Configuration ---
st.set_page_config(layout="wide", page_title="XHire: AI-Powered CV Screening Tool")

# --- Session State Initialization ---
if 'candidates_df' not in st.session_state:
    st.session_state.candidates_df = pd.DataFrame(columns=[
        'File Name', 'Match Score', 'Extracted Skills', 'Demographic Mock', 'XAI Explanation'
    ])
if 'job_profile' not in st.session_state:
    st.session_state.job_profile = DEFAULT_JOB_PROFILE

# --- Helper Functions ---

def process_resume(uploaded_file, job_profile):
    """Processes a single uploaded resume."""
    st.info(f"Processing {uploaded_file.name}...")
    
    # 1. Text Extraction
    raw_text = extract_text(uploaded_file)
    
    # 2. Skill Extraction
    extracted_skills = extract_skills(raw_text)
    
    # 3. Scoring
    match_score = calculate_score(extracted_skills, job_profile)
    
    # 4. XAI Explanation
    xai_explanation = explain_score_lime(extracted_skills, job_profile)
    
    # Mock data for bias detection and 3D visualization
    demographic_mock = random.choice(["Group A", "Group B", "Group C"])
    skill_count = len(extracted_skills)
    
    new_candidate = {
        'File Name': uploaded_file.name,
        'Match Score': match_score,
        'Extracted Skills': ", ".join(extracted_skills),
        'Demographic Mock': demographic_mock,
        'XAI Explanation': xai_explanation,
        'Skill Count': skill_count,
        'Experience Mock': random.randint(1, 15) # Mock experience for 3D plot
    }
    
    return new_candidate

def handle_upload():
    """Handles the file upload and processing for all files."""
    uploaded_files = st.session_state.uploader
    if uploaded_files:
        # Clear previous results if new files are uploaded
        st.session_state.candidates_df = pd.DataFrame(columns=[
            'File Name', 'Match Score', 'Extracted Skills', 'Demographic Mock', 'XAI Explanation', 'Skill Count', 'Experience Mock'
        ])
        
        progress_bar = st.progress(0)
        for i, file in enumerate(uploaded_files):
            new_candidate = process_resume(file, st.session_state.job_profile)
            st.session_state.candidates_df.loc[len(st.session_state.candidates_df)] = new_candidate
            progress_bar.progress((i + 1) / len(uploaded_files))
        
        st.success(f"Successfully processed {len(uploaded_files)} resumes!")
        progress_bar.empty()

# --- UI Components ---

def display_3d_visualization(df):
    """Creates and displays the 3D scatter plot."""
    st.header("3D Candidate Landscape (Attractive Figure)")
    
    if df.empty:
        st.warning("Upload resumes to see the 3D visualization.")
        return

    # Use Plotly Express for an interactive 3D scatter plot
    fig = px.scatter_3d(
        df, 
        x='Match Score', 
        y='Skill Count', 
        z='Experience Mock',
        color='Demographic Mock',
        size='Match Score',
        hover_data=['File Name'],
        title="Candidate Clustering by Score, Skills, and Experience",
        height=600
    )
    
    fig.update_layout(
        scene = dict(
            xaxis_title='Match Score (0-100)',
            yaxis_title='Skill Count',
            zaxis_title='Experience (Years)'
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_results_table(df):
    """Displays the main results table and handles XAI/Bias details."""
    st.header("Candidate Ranking and Details")
    
    if df.empty:
        st.info("No candidates processed yet.")
        return

    # Sort by score
    df_display = df.sort_values(by='Match Score', ascending=False).reset_index(drop=True)
    df_display.index = df_display.index + 1 # 1-based indexing for ranking

    # Display the main table
    st.dataframe(
        df_display[['File Name', 'Match Score', 'Extracted Skills', 'Demographic Mock']],
        use_container_width=True
    )

    # Bias Detection
    st.subheader("Bias Detection Report")
    bias_report = detect_bias(df_display)
    st.json(bias_report)
    
    # XAI Explanation Viewer
    st.subheader("Explainable AI (XAI) Viewer")
    selected_file = st.selectbox(
        "Select a candidate to view XAI explanation:",
        df_display['File Name']
    )
    
    if selected_file:
        explanation = df_display[df_display['File Name'] == selected_file]['XAI Explanation'].iloc[0]
        st.json(explanation)
        st.markdown(
            "**LIME Explanation:** The values represent the contribution of each skill to the final score. Positive values increase the score, negative values decrease it."
        )

# --- Main Application Layout ---

st.title("XHire: AI-Powered CV Screening Tool")
st.markdown("---")

# Sidebar for Job Profile Configuration
with st.sidebar:
    st.header("Job Profile Configuration")
    st.markdown("Define the required skills and their importance (weight) for the job.")
    
    new_profile = {}
    for skill, weight in st.session_state.job_profile.items():
        new_profile[skill] = st.slider(f"Weight for **{skill}**", 1, 10, weight)
        
    if st.button("Update Job Profile"):
        st.session_state.job_profile = new_profile
        st.success("Job Profile Updated! Re-upload files to apply new weights.")
        
    st.markdown("---")
    st.header("Upload Resumes")
    st.file_uploader(
        "Upload PDF or DOCX files",
        type=['pdf', 'docx'],
        accept_multiple_files=True,
        key='uploader',
        on_change=handle_upload
    )

# Main Content Tabs
tab1, tab2, tab3 = st.tabs(["Dashboard", "Raw Data & XAI", "AI Chatbot (Coming Soon)"])

with tab1:
    st.header("Screening Dashboard")
    
    # Display key metrics
    if not st.session_state.candidates_df.empty:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Candidates", len(st.session_state.candidates_df))
        col2.metric("Highest Score", st.session_state.candidates_df['Match Score'].max())
        col3.metric("Average Score", f"{st.session_state.candidates_df['Match Score'].mean():.2f}")
        
    display_3d_visualization(st.session_state.candidates_df)

with tab2:
    display_results_table(st.session_state.candidates_df)

with tab3:
    st.header("Interactive AI Chatbot Assistant")
    from openai import OpenAI

    # Initialize OpenAI client
    client = OpenAI()

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

    if prompt := st.chat_input('Ask a question about the candidates...'):
        st.session_state.messages.append({'role': 'user', 'content': prompt})
        with st.chat_message('user'):
            st.markdown(prompt)

        with st.chat_message('assistant'):
            message_placeholder = st.empty()
            full_response = ''
            
            # Create a context string from the candidates dataframe
            context = ''
            if not st.session_state.candidates_df.empty:
                context = st.session_state.candidates_df.to_string()

            # Call the OpenAI API
            try:
                stream = client.chat.completions.create(
                    model='gpt-3.5-turbo',
                    messages=[
                        {'role': 'system', 'content': 'You are a helpful HR assistant. Use the following data to answer questions about the candidates. The data is in a pandas DataFrame string format.'},
                        {'role': 'user', 'content': f'Here is the candidate data:\n{context}'},
                        {'role': 'user', 'content': prompt}
                    ],
                    stream=True,
                )
                for chunk in stream:
                    full_response += (chunk.choices[0].delta.content or '')
                    message_placeholder.markdown(full_response + '▌')
                message_placeholder.markdown(full_response)
            except Exception as e:
                full_response = f'An error occurred: {e}'
                message_placeholder.markdown(full_response)

        st.session_state.messages.append({'role': 'assistant', 'content': full_response})

# --- Initial Run Mock Data ---
if st.session_state.candidates_df.empty:
    st.info("For demonstration, mock data is generated. Upload your own files to see real processing.")
    
    # Generate mock data for the 3D visualization to be visible on first load
    mock_data = []
    for i in range(5):
        skills = random.sample(list(DEFAULT_JOB_PROFILE.keys()), k=random.randint(3, 7))
        score = calculate_score(skills, DEFAULT_JOB_PROFILE)
        mock_data.append({
            'File Name': f"Mock_Candidate_{i+1}.pdf",
            'Match Score': score,
            'Extracted Skills': ", ".join(skills),
            'Demographic Mock': random.choice(["Group A", "Group B", "Group C"]),
            'XAI Explanation': explain_score_lime(skills, DEFAULT_JOB_PROFILE),
            'Skill Count': len(skills),
            'Experience Mock': random.randint(1, 15)
        })
    
    st.session_state.candidates_df = pd.DataFrame(mock_data)
    # st.experimental_rerun() # Rerun is not available in this environment, but the state is set.
