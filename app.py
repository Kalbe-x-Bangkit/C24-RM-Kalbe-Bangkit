import os
import subprocess
from PIL import Image
import io
import gradio as gr
from transformers import AutoProcessor, TextIteratorStreamer
from transformers import Idefics2ForConditionalGeneration
import torch
from peft import LoraConfig
from transformers import AutoProcessor, BitsAndBytesConfig, IdeficsForVisionText2Text

# Baca konten HTML dari file index.html
with open('index.html', encoding='utf-8') as file:
    html_content = file.read()

DEVICE = torch.device("cuda")

USE_LORA = False
USE_QLORA = True

if USE_QLORA or USE_LORA:
    lora_config = LoraConfig(
        r=8,
        lora_alpha=8,
        lora_dropout=0.1,
        target_modules='.*(text_model|modality_projection|perceiver_resampler).*(down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj).*$',
        use_dora=False if USE_QLORA else True,
        init_lora_weights="gaussian"
    )
    if USE_QLORA:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )

    # Model yang akan digunakan
    model = Idefics2ForConditionalGeneration.from_pretrained(
        "jihadzakki/idefics2-8b-vqarad-delta",
        torch_dtype=torch.float16,
        quantization_config=bnb_config
    )

processor = AutoProcessor.from_pretrained(
    "HuggingFaceM4/idefics2-8b",
)

def format_answer(image, question, history):
    try:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question}
                ]
            }
        ]

        text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=[text.strip()], images=[image], return_tensors="pt", padding=True)
        inputs = {key: value.to(DEVICE) for key, value in inputs.items()}
        generated_ids = model.generate(**inputs, max_new_tokens=64)
        generated_texts = processor.batch_decode(generated_ids[:, inputs["input_ids"].size(1):], skip_special_tokens=True)[0]

        history.append((image, f"Question: {question} | Answer: {generated_texts}"))

        # Store the predicted answer in a variable before deleting intermediate variables
        predicted_answer = f"Predicted Answer: {generated_texts}"

        # Clear the cache and delete unnecessary variables
        del inputs
        del generated_ids
        del generated_texts
        torch.cuda.empty_cache()

        return predicted_answer, history
    except Exception as e:
        # Clear the cache in case of an error
        torch.cuda.empty_cache()
        return f"Error: {str(e)}", history

def clear_history():
    return "", []

def undo_last(history):
    if history:
        history.pop()
    return "", history

def retry_last(image, question, history):
    if history:
        last_image, last_entry = history[-1]
        return format_answer(last_image, question, history[:-1])
    return "No previous analysis to retry.", history

def switch_theme(mode):
    if mode == "Light Mode":
        return gr.themes.Default()
    else:
        return gr.themes.Soft(primary_hue=gr.themes.colors.green, secondary_hue=gr.themes.colors.green)

def save_feedback(feedback):
    return "Thank you for your feedback!"

def display_history(history):
    log_entries = []
    for img, text in history:
        log_entries.append((img, text))
    return log_entries

# Build the Visual QA application using Gradio with improvements
with gr.Blocks(
    theme=gr.themes.Soft(
        font=[gr.themes.GoogleFont("Inconsolata"), "Arial", "sans-serif"],
        primary_hue=gr.themes.colors.green,
        secondary_hue=gr.themes.colors.green,
    )
) as VisualQAApp:
    gr.HTML(html_content)  # Display the HTML content

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(label="Upload image", type="pil")
        with gr.Column(scale=1):
            with gr.Column(scale=1):
                question_input = gr.Textbox(show_label=False, placeholder="Enter your question here...", lines=2)
                submit_button = gr.Button("Submit", variant="primary")
                answer_output = gr.Textbox(label="Result Prediction", lines=2)

    history_state = gr.State([])  # Initialize the history state

    submit_button.click(
        format_answer,
        inputs=[image_input, question_input, history_state],
        outputs=[answer_output, history_state],
        show_progress=True
    )

    with gr.Row():
        retry_button = gr.Button("🔄 Retry")
        undo_button = gr.Button("↩️ Undo")
        clear_button = gr.Button("🗑️ Clear")

        retry_button.click(
            retry_last,
            inputs=[image_input, question_input, history_state],
            outputs=[answer_output, history_state]
        )

        undo_button.click(
            undo_last,
            inputs=[history_state],
            outputs=[answer_output, history_state]
        )

        clear_button.click(
            clear_history,
            inputs=[],
            outputs=[answer_output, history_state]
        )

    with gr.Row():
        history_gallery = gr.Gallery(label="History Log", elem_id="history_log")
        submit_button.click(
            display_history,
            inputs=[history_state],
            outputs=[history_gallery]
        )

    gr.Markdown("## Contoh Input dengan Teks")
    with gr.Row():
        with gr.Column():
            gr.Examples(
                examples=[
                    ["sample_data/images/Gambar-Otak-Slake.jpg", "What modality is used to take this image?"],
                    ["sample_data/images/Gambar-Otak-Slake2.jpg", "Which part of the body does this image belong to?"]
                ],
                inputs=[image_input, question_input],
                outputs=[answer_output, history_state],
                label="Upload image",
                elem_id="Prompt"
            )

    with gr.Accordion("Help", open=False):
        gr.Markdown("**Upload image**: Select the chest X-ray image you want to analyze.")
        gr.Markdown("**Enter your question**: Type the question you have about the image, such as 'What modality is used to take this image?'")
        gr.Markdown("**Submit**: Click the submit button to get the prediction from the model.")

    with gr.Accordion("User Preferences", open=False):
        gr.Markdown("**Mode**: Choose between light and dark mode for your comfort.")
        mode_selector = gr.Radio(choices=["Light Mode", "Dark Mode"], label="Select Mode")
        apply_theme_button = gr.Button("Apply Theme")

        apply_theme_button.click(
            switch_theme,
            inputs=[mode_selector],
            outputs=[],
        )

    with gr.Accordion("Feedback", open=False):
        gr.Markdown("**We value your feedback!** Please provide any feedback you have about this application.")
        feedback_input = gr.Textbox(label="Feedback", lines=4)
        submit_feedback_button = gr.Button("Submit Feedback")

        submit_feedback_button.click(
            save_feedback,
            inputs=[feedback_input],
            outputs=[feedback_input]
        )

VisualQAApp.launch(share=True, debug=True)
