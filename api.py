import gradio as gr
import soundfile as sf
from TTS.api import TTS
import json

from cleantext import clean
# Load VITS model
class MyTTS(TTS):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._is_multi_lingual = False  # Use a private attribute for storage

    # Property getter for is_multi_lingual
    @property
    def is_multi_lingual(self):
        return self._is_multi_lingual

    # Property setter for is_multi_lingual
    @is_multi_lingual.setter
    def is_multi_lingual(self, value):
        self._is_multi_lingual = value



def load_tts_model(model_path, config_path):
    """Load the TTS model with the updated config."""
    try:
        tts_model = MyTTS(model_path=model_path, config_path=config_path)
        print("TTS model loaded successfully.")
        print("Model config after loading: ", tts_model.config)
        return tts_model
    except Exception as e:
        print(f"Error loading TTS model: {e}")
        return None
#tts = load_tts_model(model_path="arthur_morgan_ai_voice/rdr_vits_final_model.pth", config_path="arthur_morgan_ai_voice/rdr_vits_final_config.json")

def normalize(text):
    text = clean(text,
                 fix_unicode=True,       # Fix unicode such as smart quotes
                 to_ascii=True,           # Transliterate to closest ASCII representation
                 lower=True,              # Convert text to lowercase
                 no_urls=True,            # Remove URLs
                 no_emails=True,          # Remove emails
                 no_phone_numbers=True,   # Remove phone numbers
                 no_currency_symbols=True # Remove currency symbols
                 )
    return text


def update_config(length_scale, inference_noise_scale, inference_noise_scale_dp, config_path):
    """Update the config file with the dynamic parameters."""
    
    with open(config_path, "r+") as f:
        config = json.load(f)
        config["model_args"]["length_scale"] = length_scale
        config["model_args"]["inference_noise_scale"] = inference_noise_scale
        config["model_args"]["inference_noise_scale_dp"] = inference_noise_scale_dp
        f.seek(0)
        json.dump(config, f, indent=4)
        f.truncate()
        f.flush()



def text_to_speech(text, length_scale=898.0, inference_noise_scale=0.6, inference_noise_scale_dp=0.8, format='wav', normalize_text=False):
    config_path = "config.json"
    model_path = "model.pth"

    # Update config file with the parameters
    update_config(length_scale, inference_noise_scale, inference_noise_scale_dp, config_path)
    

    if 'tts' in globals():
        del tts  # Clear the old model instance

    # Reload the TTS model with the updated config
    tts = load_tts_model(model_path, config_path)

    if normalize_text:
        text = normalize(text)

    try:
        # Generate the audio output
        audio_output = tts.tts(text)
    except Exception as e:
        print(f"Error generating audio: {e}")
        return None

    # Save the output file in the specified format
    output_file = f"output.{format}"
    sf.write(output_file, audio_output, samplerate=22050)
    
    return output_file



# Test the text_to_speech function without Gradio
# if __name__ == "__main__":
#     output_file = text_to_speech(
#         text="Hello, there is so much I want to tell you.",
#         length_scale= 0.5,  # Adjust these values to see different effects
#         inference_noise_scale=0.8,
#         inference_noise_scale_dp=1.0,
#         format='wav',
#         normalize_text=True
#     )
#     print(f"Audio saved to {output_file}")



demo = gr.Blocks()

with demo: 
    
    gr.Markdown("# Arthur Morgan VITS Model")
    gr.Markdown("""# This is my attempt at voice cloning 
                using the Coqui TTS VITS model. I have used voice 
                lines available [here](insert link). The original voice belongs to Roger Clark, 
                who portrayed Arthur Morgan in the Red Dead Redemption II game. Below, you can 
                experiment with different inference settings and try out the synthesized voice for yourself.
                I've also attached audio snippets of the dataset I used to train this model.
                Feel free to explore and let me know your thoughts!
                Thank you for checking it out!""")
    with gr.Row(): 
        with gr.Column(scale = 2 , min_width = '150'): 
            gr.Image(value = "image.png", show_label = False, width = 700, height = 350) 
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(label="Input Text", placeholder="Enter text here...")
            length_scale = gr.Slider(minimum=0, maximum=2, step=0.01, value=1, label="Length Scale")
            inference_noise_scale = gr.Slider(minimum=0, maximum=2, step=0.01, value=0.5, label="Inference Noise Scale")
            inference_noise_scale_dp = gr.Slider(minimum=0, maximum=2, step=0.01, value=0.3, label="Inference Noise Scale DP")
            audio_format = gr.Dropdown(choices=["wav", "mp3"], value="wav", label="Audio Format")
            normalize_text = gr.Checkbox(value=True, label="Enable Text Normalization")
        
        # Output area for generated audio
        output_audio = gr.Audio(type="filepath", label="Generated Audio")
    
    # Define the submit button and clear button
    submit_button = gr.Button("Submit")
    clear_button = gr.Button("Clear")
    # Add example audio tracks below the input components
    gr.Markdown("### Example Arthur Morgan's Voice:")
    gr.Audio("example_audio_1.mp3", label="Example of original voice #1: It's easier to think you are tougher than you are with a gun in your hand.")
    gr.Audio("example_audio_2.mp3", label="Example of original voice #2: Make yourself useful. Oh, Look at you.")
    gr.Audio("example_audio_3.mp3", label="Example of original voice #3: What are you trying to do? Brush the dirt off?")



    # Function to trigger on submit button
    submit_button.click(fn=text_to_speech, inputs=[input_text, length_scale, inference_noise_scale, inference_noise_scale_dp, audio_format, normalize_text], outputs=output_audio)

# Launch the Gradio interface

print('Starting the demo...')
demo.launch(share = True)





