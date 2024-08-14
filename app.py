import streamlit as st
from transformers import pipeline
from diffusers import StableDiffusionPipeline
from gtts import gTTS
from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips
import torch
import gc
from multiprocessing import Pool

# Load Hugging Face models
text_generator = pipeline('text-generation', model='gpt2')
image_generator = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float32)
# Ensure model is on CPU
image_generator = image_generator.to("cpu")

def generate_scene_descriptions(script):
    scenes = text_generator(script, max_length=100, num_return_sequences=1)
    return scenes[0]['generated_text'].split('\n\n')  # Split into scenes

def generate_image_with_stable_diffusion(description):
    image = image_generator(description).images[0]
    filename = f"{description[:10].replace(' ', '_')}.png"
    image.save(filename)
    return filename

def generate_images(scene_descriptions):
    with Pool() as pool:
        images = pool.map(generate_image_with_stable_diffusion, scene_descriptions)
    return images

def generate_voiceover(text, filename):
    tts = gTTS(text)
    tts.save(filename)

def compile_video(images, voiceovers):
    clips = []
    for image, voiceover in zip(images, voiceovers):
        img_clip = ImageClip(image).set_duration(5)  # Assuming each voiceover is 5 seconds
        audio_clip = AudioFileClip(voiceover)
        video_clip = img_clip.set_audio(audio_clip)
        clips.append(video_clip)

    final_video = concatenate_videoclips(clips)
    final_video.write_videofile("output_video.mp4")

st.title("Text-to-Video Generator")
uploaded_file = st.file_uploader("Upload your script", type=["txt"])
script = st.text_area("Or paste your script here")

if st.button("Generate Video"):
    if uploaded_file:
        script = uploaded_file.read().decode('utf-8')
    if script:
        scenes = generate_scene_descriptions(script)
        images = generate_images(scenes)
        voiceovers = []
        for i, scene in enumerate(scenes):
            filename = f"voiceover_{i}.mp3"
            generate_voiceover(scene, filename)
            voiceovers.append(filename)
        compile_video(images, voiceovers)
        st.success("Video generated successfully!")
    else:
        st.error("Please upload or paste a script.")
    # Clear memory
    gc.collect()
