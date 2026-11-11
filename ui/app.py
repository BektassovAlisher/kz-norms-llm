"""Gradio-демо: вопрос → ответы base и fine-tuned рядом (наполним позже)."""

import gradio as gr

with gr.Blocks(title="Нормы РК — ассистент") as demo:
    gr.Markdown("# Ассистент по строительным нормам РК\n_Не является экспертным заключением._")
    question = gr.Textbox(label="Вопрос")
    with gr.Row():
        base_out = gr.Markdown(label="base")
        ft_out = gr.Markdown(label="fine-tuned")

if __name__ == "__main__":
    demo.launch()
