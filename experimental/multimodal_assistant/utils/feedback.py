import gspread
import datetime
import streamlit as st

def add_row_to_sheet(values):
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    gc = gspread.service_account(filename="service.json")

    sh = gc.open_by_url("https://docs.google.com/spreadsheets/d/1R8sDCJ2jBSvEKh4awAOXgfhJIaftEtTBhP2SuDxutD4/edit#gid=930743170")

    worksheet = sh.get_worksheet(0)

    worksheet.append_row(values)

def submit_feedback(feedback, query, response):
    face = feedback['score']
    score = {"😀": 5, "🙂": 4, "😐": 3, "🙁": 2, "😞": 1}[face]
    comment = "Score: " + str(score) + "/5. Feedback: " + feedback['text'] or "none"
    feedback_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    feedback_data = (feedback_time, score, query, response, comment)
    st.toast("Thank you for your feedback!")
    # add to Google Sheet here
    try:
        add_row_to_sheet(feedback_data)
    except Exception as e:
        st.toast(f"Failed to add to Google Sheet. Exception: {e}")

feedback_kwargs = {
        "feedback_type": "faces",
        "optional_text_label": "Please provide feedback",
        "on_submit": submit_feedback,
    }