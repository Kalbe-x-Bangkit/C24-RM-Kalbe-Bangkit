import gradio as gr
from transformers import pipeline
import plotly.graph_objs as go

# Load the Visual QA model
generator = pipeline("visual-question-answering", model="jihadzakki/blip1-medvqa")

def format_answer(image, question, history):
    try:
        result = generator(image, question, max_new_tokens=50)
        predicted_answer = result[0].get('answer', 'No answer found')
        history.append(f"Question: {question} | Answer: {predicted_answer}")
        # Create a simple chart for demonstration purposes
        chart = create_chart(predicted_answer)
        return f"Predicted Answer: {predicted_answer}", chart, history
    except Exception as e:
        return f"Error: {str(e)}", None, history

def create_chart(predicted_answer):
    # Example chart data
    labels = ['Positive', 'Negative']
    values = [1, 0] if predicted_answer.lower() == 'yes' else [0, 1]
    colors = ['blue', 'red'] if predicted_answer.lower() == 'yes' else ['red', 'blue']

    fig = go.Figure(data=[go.Pie(labels=labels, values=values, marker=dict(colors=colors))])
    return fig

def switch_theme(mode):
    if mode == "Light Mode":
        return gr.themes.Default()
    else:
        return gr.themes.Soft(primary_hue=gr.themes.colors.blue, secondary_hue=gr.themes.colors.orange)

def save_feedback(feedback):
    return "Thank you for your feedback!"

# Build the Visual QA application using Gradio with improvements
with gr.Blocks(
    theme=gr.themes.Soft(
        font=[gr.themes.GoogleFont("Inconsolata"), "Arial", "sans-serif"],
        primary_hue=gr.themes.colors.blue,
        secondary_hue=gr.themes.colors.red,
    )
) as VisualQAApp:
    gr.Markdown("# Visual Question Answering using BLIP Model", elem_classes="title")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(label="Upload image", type="pil")
            question_input = gr.Textbox(show_label=False, placeholder="Enter your question here...")
            submit_button = gr.Button("Submit", variant="primary")

        with gr.Column():
            answer_output = gr.Textbox(label="Result Prediction")
            chart_output = gr.Plot(label="Interactive Chart")

    history_state = gr.State([])  # Initialize the history state

    submit_button.click(
        format_answer,
        inputs=[image_input, question_input, history_state],
        outputs=[answer_output, chart_output, history_state],
        show_progress=True
    )

    with gr.Row():
        history = gr.Textbox(label="History Log", lines=10)
        gr.Markdown("**Log of previous interactions:**")
        submit_button.click(
            lambda history: "\n".join(history),
            inputs=[history_state],
            outputs=[history]
        )

    with gr.Accordion("Help", open=False):
        gr.Markdown("**Upload image**: Select the chest X-ray image you want to analyze.")
        gr.Markdown("**Enter your question**: Type the question you have about the image, such as 'Is there any sign of pneumonia?'")
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

VisualQAApp.launch(share=True, server_name="0.0.0.0", server_port=8080, debug=True)


