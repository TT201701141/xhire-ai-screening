import spacy
import re
from io import StringIO, BytesIO
from pdfminer.high_level import extract_text_to_fp
from docx import Document
from lime.lime_tabular import LimeTabularExplainer
import numpy as np
import pandas as pd

# Load spaCy model once
try:
    nlp = spacy.load("en_core_web_sm")
except:
    print("SpaCy model 'en_core_web_sm' not found. Please run 'python -m spacy download en_core_web_sm'")
    nlp = None

# --- 1. Text Extraction ---

def extract_text_from_pdf(file_stream):
    """Extracts text from a PDF file stream."""
    output_string = StringIO()
    extract_text_to_fp(file_stream, output_string)
    return output_string.getvalue()

def extract_text_from_docx(file_stream):
    """Extracts text from a DOCX file stream."""
    document = Document(file_stream)
    return "\n".join([paragraph.text for paragraph in document.paragraphs])

def extract_text(uploaded_file):
    """Routes file to the correct text extraction function."""
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    # Read the file content once
    file_stream = uploaded_file.read()
    
    if file_extension == 'pdf':
        # For PDF, we need a BytesIO object with the raw bytes
        return extract_text_from_pdf(BytesIO(file_stream))
    elif file_extension == 'docx':
        # For DOCX, we need a BytesIO object
        return extract_text_from_docx(BytesIO(file_stream))
    else:
        return "Unsupported file type."

# --- 2. Skill Matching and Extraction ---

# Mock list of skills for demonstration
MOCK_SKILLS = [
    "Python", "Streamlit", "NLP", "Machine Learning", "Deep Learning", 
    "Data Analysis", "SQL", "Cloud Computing", "Project Management", 
    "Communication", "Teamwork", "Leadership", "Agile", "Scrum"
]

def extract_skills(text):
    """
    Extracts skills from text using a simple keyword matching approach.
    In a real-world scenario, this would be a more sophisticated NER model.
    """
    if not nlp:
        return []
        
    found_skills = set()
    text_lower = text.lower()
    
    for skill in MOCK_SKILLS:
        if skill.lower() in text_lower:
            found_skills.add(skill)
            
    return list(found_skills)

# --- 3. Candidate Scoring ---

def calculate_score(extracted_skills, job_profile):
    """
    Calculates a weighted score for a candidate based on a job profile.
    
    :param extracted_skills: List of skills found in the resume.
    :param job_profile: Dictionary with required skills and their weights.
    :return: Total score (float).
    """
    score = 0
    total_possible_score = sum(job_profile.values())
    
    extracted_skills_lower = {skill.lower() for skill in extracted_skills}
    
    for skill, weight in job_profile.items():
        if skill.lower() in extracted_skills_lower:
            score += weight
            
    # Normalize score to be between 0 and 100
    if total_possible_score > 0:
        normalized_score = (score / total_possible_score) * 100
    else:
        normalized_score = 0
        
    return round(normalized_score, 2)

# --- Mock Job Profile for Demonstration ---
DEFAULT_JOB_PROFILE = {
    "Python": 5,
    "Streamlit": 4,
    "NLP": 5,
    "Machine Learning": 4,
    "Data Analysis": 3,
    "Communication": 2,
    "Teamwork": 2
}

# --- 4. Explainable AI (XAI) using LIME ---

def get_feature_vector(extracted_skills, job_profile):
    """Converts skills to a feature vector for LIME."""
    all_skills = list(job_profile.keys())
    feature_vector = [1 if skill.lower() in {s.lower() for s in extracted_skills} else 0 for skill in all_skills]
    return np.array(feature_vector)

def predict_fn(data, job_profile):
    """Prediction function for LIME (returns score as a probability-like value)."""
    all_skills = list(job_profile.keys())
    predictions = []
    for row in data:
        # Reconstruct skills from the feature vector (1s indicate presence)
        extracted_skills = [all_skills[i] for i, val in enumerate(row) if val == 1]
        score = calculate_score(extracted_skills, job_profile)
        # LIME expects probabilities or a single class prediction. 
        # We'll return the normalized score (0-100) divided by 100 as a regression output.
        predictions.append(score / 100.0) 
    return np.array(predictions)

def explain_score_lime(extracted_skills, job_profile):
    """Generates LIME explanation for a candidate's score."""
    all_skills = list(job_profile.keys())
    
    # Create a dummy training set for LIME (100 random samples)
    np.random.seed(42)
    training_data = np.random.randint(0, 2, size=(100, len(all_skills)))
    
    explainer = LimeTabularExplainer(
        training_data=training_data,
        feature_names=all_skills,
        class_names=['Score'],
        mode='regression',
        random_state=42
    )
    
    feature_vector = get_feature_vector(extracted_skills, job_profile)
    
    # Wrap predict_fn to pass job_profile
    predict_wrapper = lambda x: predict_fn(x, job_profile)
    
    explanation = explainer.explain_instance(
        data_row=feature_vector,
        predict_fn=predict_wrapper,
        num_features=len(all_skills)
    )
    
    # Convert explanation to a dictionary for easy display
    explanation_dict = {feature: weight for feature, weight in explanation.as_list()}
    return explanation_dict

# --- 5. Bias Detection (Mock) ---

def detect_bias(candidates_df):
    """
    Mocks a simple bias detection by comparing average scores across a mock demographic feature.
    
    :param candidates_df: DataFrame of candidates with 'Match Score' and 'Demographic Mock' columns.
    :return: Dictionary of average scores per group.
    """
    if candidates_df.empty or 'Demographic Mock' not in candidates_df.columns:
        return {"Error": "No data or 'Demographic Mock' column missing."}
        
    avg_scores = candidates_df.groupby('Demographic Mock')['Match Score'].mean().to_dict()
    
    # Simple check for score disparity
    if len(avg_scores) >= 2:
        # Find the max and min average scores
        max_group = max(avg_scores, key=avg_scores.get)
        min_group = min(avg_scores, key=avg_scores.get)
        max_score = avg_scores[max_group]
        min_score = avg_scores[min_group]
        disparity = max_score - min_score
        
        if disparity > 10: # Arbitrary threshold for "bias"
            avg_scores['Bias_Alert'] = f"Significant score disparity ({disparity:.2f}) between {max_group} (High) and {min_group} (Low)."
        else:
            avg_scores['Bias_Alert'] = "No significant score disparity detected."
            
    return avg_scores
