import gradio as gr
import torch
from diffusers import StableDiffusion3Pipeline

model_id = "stabilityai/stable-diffusion-3.5-medium"
device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = StableDiffusion3Pipeline.from_pretrained(
    model_id,
    dtype=torch.float16 if device == "cuda" else torch.float32,
    text_encoder_3=None,
    tokenizer_3=None,
)

if device == "cuda":
    pipe.enable_model_cpu_offload()
else:
    pipe.to("cpu")


def image_generation(prompt):
    image = pipe(
        prompt=prompt,
        negative_prompt="blurred, ugly, watermark, low resolution, blurry",
        num_inference_steps=40,
        height=1024,
        width=1024,
        guidance_scale=9.0,
    ).images[0]
    return image


interface = gr.Interface(
    fn=image_generation,
    inputs=gr.Textbox(lines=2, placeholder="Enter your prompt..."),
    outputs=gr.Image(type="pil"),
    title="GENAI PROJECT 9: Image Creation Using Stable Diffusion 3.5",
    description="Generate awesome images using the Stable Diffusion 3.5 model.",
)

interface.launch()