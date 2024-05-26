import gradio as gr
from transformers import pipeline
import time

generator = pipeline("visual-question-answering", model="jihadzakki/blip1-medvqa")

def format_answer(image, question):
    result = generator(image, question)
    print(result)  # Print the result to see its structure
    predicted_answer = result[0].get('answer', 'No answer found')
    return f"Predicted Answer: {predicted_answer}"

def predict(image, question, progress=gr.Progress()):
    progress(0, desc="Processing...")
    # Simulate the processing steps with a sleep
    time.sleep(1)
    progress(50, desc="Generating answer...")
    result = format_answer(image, question)
    progress(100, desc="Done")
    return result

# Build the Visual QA application using Gradio with a theme
with gr.Blocks(
    theme=gr.themes.Soft(
        font=[gr.themes.GoogleFont("Inconsolata"), "Arial", "sans-serif"]
        )
    ) as VisualQAApp:
    gr.Markdown("# Visual Question Answering using BLIP Model")
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(label="Upload image", type="pil")
            question_input = gr.Textbox(show_label=False, placeholder="Enter your question here...")
            submit_button = gr.Button("Submit", variant="primary")
        with gr.Column():
            answer_output = gr.Textbox(label="Result Prediction")

    submit_button.click(
        predict,
        inputs=[image_input, question_input],
        outputs=[answer_output]
    )

VisualQAApp.launch(share=True)
