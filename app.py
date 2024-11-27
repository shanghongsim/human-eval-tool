import streamlit as st
import json
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime
from nltk import sent_tokenize
import re
from fuzzywuzzy import fuzz
import string
import nltk
nltk.download('punkt_tab')

# Thresholds and flags
REJECTION_FUZZ_THRESHOLD=85
REJECTION_FLAG="I apologize, but I couldn't find an answer"

# Initialize Firebase Admin SDK
def initialize_firebase():
    # cred = credentials.Certificate("YOUR/PATH/TO/firebase-adminsdk-XXXXXX-XXXXXXXXXX.json") # Call once on creation
    service_account_info = {
        "type": st.secrets["firebase"]["type"],
        "project_id": st.secrets["firebase"]["project_id"],
        "private_key_id": st.secrets["firebase"]["private_key_id"],
        "private_key": st.secrets["firebase"]["private_key"].replace("\\n", "\n"),
        "client_email": st.secrets["firebase"]["client_email"],
        "client_id": st.secrets["firebase"]["client_id"],
        "auth_uri": st.secrets["firebase"]["auth_uri"],
        "token_uri": st.secrets["firebase"]["token_uri"],
        "auth_provider_x509_cert_url": st.secrets["firebase"]["auth_provider_x509_cert_url"],
        "client_x509_cert_url": st.secrets["firebase"]["client_x509_cert_url"]
    }
    if not firebase_admin._apps:
        cred = credentials.Certificate(service_account_info)
        firebase_admin.initialize_app(cred)
    return firestore.client()

# Initialize Firestore
db = initialize_firebase()

# Utility Functions
def escape_markdown(text):
    import re
    escape_chars = r'([\[\](){}*+?.\\^$|#])'
    return re.sub(escape_chars, r'\\\1', text)

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def load_raw_data():
    with open('human_eval_asqa_mix.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def get_data(form_number, chunk_size=25):
    raw_data = load_raw_data()
    forms = [raw_data[i:i + chunk_size] for i in range(0, len(raw_data), chunk_size)]
    return forms[form_number - 1] # Adjust for zero-based indexing

def save_responses_to_firestore(responses, form_number, code, name):
    collection_ref = db.collection(f'{name}_{code}')
    for question, response in responses.items():
        doc_ref = collection_ref.document()
        doc_ref.set({
            'timestamp': datetime.now().isoformat(),
            'question_set': code,
            'rater': name,
            'question': question,
            'response': response
        })

# Form display (based on form_number)
def display_form(form_number, code, name):
    st.header(f"Set #{form_number}")

    data = get_data(form_number, chunk_size=25)
    
    with st.form(key=f'form_{form_number}'):
        responses = {}
        
        for sample_idx, sample in enumerate(data):
            st.subheader(f"Sample {sample_idx + 1}: ")
            st.markdown(f"**Question:** \"{escape_markdown(sample['question'])}\"")
            
            for idx, doc in enumerate(sample['docs']):
                st.markdown(f"- **Document [{idx + 1}]** (Title: {escape_markdown(doc['title'])}): {escape_markdown(doc['text'])}")

            # likert_options = ["5 - Strongly Agree", "4 - Agree", "3 - Neutral", "2 - Disagree", "1 - Strongly Disagree"]
            correctness_options = ["Correct", "Wrong", "Not sure"]
            citation_recall_options = ["Full support", "No support"]
            citation_precision_options = ["Full support", "Partial support", "No support"]

            reponses = [sample['GAns'], sample['output']] # GAns is Gold Ans based on legacy naming
            responses_type = ['pos', 'neg']
            output = {}

            for resp, resp_type in zip(reponses, responses_type):
                temp_output = {}
                st.markdown(f"**Response :** \"{resp}\"")

                # Correctness rating
                correctness_rating = st.radio(f"Given the documents, the response is a correct answer to the question. You should read all five documents first to see if the response could be derived from the documents given. If the response cannot be derived, then it is necessarily also wrong. If it can be derived, then rate whether the response correctly answers the question.", 
                                              correctness_options, index=None, key=f"correctness_{resp_type}_{sample_idx}")
                temp_output['correctness'] = correctness_rating

                # Check if it's a rejection response
                is_rejection = fuzz.partial_ratio(normalize_answer(REJECTION_FLAG), normalize_answer(resp)) > REJECTION_FUZZ_THRESHOLD
                if not is_rejection:
                    sentences = sent_tokenize(resp)

                    for sent_idx, sentence in enumerate(sentences):
                        # Citation recall
                        citation_recall_rating = st.radio(f"{sentence} Does the set of citations support the claim?", 
                                                          citation_recall_options, index=None, key=f"rec_{resp_type}_{sample_idx}_{sent_idx}")
                        temp_output[f'citation_recall_sent{sent_idx}'] = citation_recall_rating

                        # Individual citation precision
                        citations = list(set(re.findall(r"\[\d+\]", sentence)))
                        if len(citations) > 1:
                            cleaned_sentence = re.sub(r"\[\d+\]", "", sentence).strip()
                            cleaned_sentence = cleaned_sentence[:-1]
                            for cite_idx, cite in enumerate(citations):
                                citation_prec_rating = st.radio(f"{cleaned_sentence} {cite}. Does this INDIVIDUAL citation support the claim?", citation_precision_options, index=None, key=f"prec_{resp_type}_{sample_idx}_{sent_idx}_{cite_idx}")
                                temp_output[f'citation_prec_sent{sent_idx}_citation{cite_idx}'] = citation_prec_rating
                
                output[resp_type] = temp_output

            st.write("---")  # Separator between samples
            responses[f"{sample['question']}"] = output

        # Form submission
        if st.form_submit_button("Submit"):
            if any(response is None for response in responses.values()):
                st.error("Please answer all questions before submitting the form.")
            else:
                st.success("Thank you for completing the survey!")
                save_responses_to_firestore(responses, form_number, code, name)
                st.write("Your responses have been recorded:")
                for q, r in responses.items():
                    st.write(f"{q}: {r}")

# Main App
def main():
    st.set_page_config(page_icon="🔬",
                       layout="wide",
                       initial_sidebar_state="expanded")
    st.title("Hallucination Human Evaluation")

    multi = """
    Thank you for participating in this human evaluation! Please review the instructions before beginning:

    1. This survey includes 10 samples. It should take no more than 20 minutes to complete.
    2. For each sample, you will see two different responses to the question. Your task is to rate each answer based on how well it addresses the question using the provided documents.
    3. The answer should be based on the information provided in the documents. If the documents do not have enough information to answer the question, a refusal is a valid response.
    4. Full support: all of the information in the statement is supported by the citation. Partial support: some of the information in the statement is supported by the citation, but other parts are not supported (e.g., missing or contradictory). No support: the citation does not support any part of the statement (e.g., the cited webpage is completely irrelevant or contradictory).
    4. Please enter your name and the access code provided in the fields below.

    Thank you for your participation!
    """
    st.markdown(multi)

    # Prompt the user to enter their code
    name = st.text_input("Enter your name:")
    code = st.text_input("Enter your access code:")

    code_to_form = { "1": 1, "2": 2, "3": 3, "4": 4} # Define codes that map to each form
    if code:
        # Validate the code
        form_number = code_to_form.get(code.upper())
        if form_number:
            st.success(f"Access granted to set #{form_number}")
            display_form(form_number, code, name)
        else:
            st.error("Invalid code. Please enter a valid access code.")
    else:
        st.info("Please enter your access code to access the survey.")


if __name__ == "__main__":
    main()
