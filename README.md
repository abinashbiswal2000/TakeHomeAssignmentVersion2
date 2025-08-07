Candidate Recommendation Engine (using streamlit)

Summary
This project helps recruiters and hiring teams automatically rank uploaded resumes against a given job description, using TF-IDF similarity scoring and AI-powered summarization to highlight top candidates.

Features
- Copy-Paste a job description
- Upload Multiple resumes (pdf, docx, txt)
- Rank resume using (TF-IDF cosine similarity, or sentence embeddings based cosine similarity)


File Structure
1 - app.py
2 - df_ai.py
3 - extractTextAndBuildCorpus.py
4 - sentenceEmbeddingsFunction.py
5 - tf_idf_Function.py


Assumptions
1 - The first words in the resume are representing the person's name.


NOTE
- There are two types of resume matching in this application
    - one uses tf-idf vectors to calculate cosine.
    - the other uses transformer based sentence enbeddings to calculate cosine.


----------------------------------------------

Details

----------------------------------------------


app.py (App structure)
- Title
- Job Description
- Resume
- Slider to choose number of resumes to display
- Buttons to calculate cosine and display table, followed by an AI summary of the best resume.


df_ai.py
- The file contains functions which make the dataframe and generate the AI summary of the best resume.


extractTextAndBuildCorpus.py
- This file contains functions which read the Job description and parse the resumes.
- The output is a list of strings, where the first string is the job description and the rest are the resumes.


sentenceEmbeddingsFunction.py
- Here there are functions to take the corpus, generate embeddings and calculate the cosine score.


tf_idf_Function.py
- Here there are functions to take the corpus, preprocess text, generate tf-idf vectors and calculate cosine.
