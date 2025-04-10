from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import pipeline

# Load pipelines
summarizer = pipeline("summarization")
qa_pipeline = pipeline("question-answering")

app = FastAPI()

# Request models
class TextInput(BaseModel):
    text: str

class QuestionInput(BaseModel):
    text: str
    question: str

@app.get("/")
def read_root():
    return {"message": "Welcome to the Text Summary & QA API"}

@app.post("/summarize")
def summarize(input: TextInput):
    summary = summarizer(input.text, max_length=150, min_length=30, do_sample=False)
    return {"summary": summary[0]["summary_text"]}

@app.post("/ask")
def ask_question(input: QuestionInput):
    answer = qa_pipeline(question=input.question, context=input.text)
    return {"answer": answer["answer"], "score": answer["score"]}

