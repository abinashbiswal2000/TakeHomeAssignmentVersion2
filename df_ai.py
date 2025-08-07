import openai
import pandas as pd
import os
import streamlit as st

try:
    api_key = st.secrets["TOGETHER_API_KEY"]
except:
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("TOGETHER_API_KEY")




def makeDataFrame (sorted_indices, resume_texts, uploaded_files, top_n):
        
        top_5 = []
        for i, score in sorted_indices[:top_n]:
            text = resume_texts[i]
            candidate_name = " ".join(text.strip().split()[:2]) if text else "Unknown"

            top_5.append({
                "Candidate Name": candidate_name,
                "Similarity Score": f"{score:.2f}",
                "File Name": uploaded_files[i].name
            })


        # Return DataFrame
        return pd.DataFrame(top_5)




def ask_together_ai(prompt):
    client = openai.OpenAI(api_key=api_key, base_url="https://api.together.xyz/v1")

    try:
        response = client.chat.completions.create(
            model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"Error: {e}"








def generatePrompt (sorted_indices, resume_texts, jd_text):
    best_index = sorted_indices[0][0]
    best_resume_text = resume_texts[best_index]

    # Prompt for AI
    return f"""Tell me why this person is a great fit for the role. Also let me know where the areas are that this person is lacking (separate paragraphs with headings). Your answer should be natural, and not more than 200 words. You may or may not use numbered lists in your response.

---------------------------
Job Description:
{jd_text}

---------------------------
Resume:
{best_resume_text}
"""