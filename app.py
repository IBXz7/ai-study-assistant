from transformers import pipeline
import gradio as gr

# --- Models ---
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

# might change the generator model to ( FLAN-T5 ) to support interactive MCQs 
generator = pipeline("text-generation", model="gpt2")

# --- Summarization ---
def summarize(text):
    result = summarizer(text, max_length=120, min_length=30, do_sample=False)
    return result[0]["summary_text"]

# --- Generate Questions ---
def generate_questions(text):
    prompt = f"""
    Based on this text, create 3 study questions:

    Text:
    {text}

    Questions:
    """
    result = generator(prompt, max_length=150, num_return_sequences=1)
    return result[0]["generated_text"]

# --- Explain Simply ---
def simplify(text):
    prompt = f"""
    Explain this in simple terms for a student:

    {text}

    Simple explanation:
    """
    result = generator(prompt, max_length=150, num_return_sequences=1)
    return result[0]["generated_text"]

# --- Full pipeline ---
def study_assistant(text):
    summary = summarize(text)
    questions = generate_questions(text)
    explanation = simplify(text)

    return summary, questions, explanation

# --- UI ---
interface = gr.Interface(
    fn=study_assistant,
    inputs=gr.Textbox(lines=12, placeholder="Write the text here ..."),
    outputs=[
        gr.Textbox(label="Summary"),
        gr.Textbox(label="Questions"),
        gr.Textbox(label="Simple Explanation")
    ],
    title="AI Study Assistant",
    description="Summarize + Generate Questions + Explain text simply"
)

interface.launch()