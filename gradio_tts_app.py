import random
import numpy as np
import torch
import gradio as gr
from chatterbox.tts import ChatterboxTTS
import io
from flask import Flask, request, jsonify, Response
import threading
import tempfile
import scipy.io.wavfile as wavfile

app = Flask(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def load_model():
    model = ChatterboxTTS.from_pretrained(DEVICE)
    return model


def generate(model, text, audio_prompt_path, exaggeration, temperature, seed_num, cfgw):
    sr, wav = generate_raw(
        model, 
        text, 
        audio_prompt_path, 
        exaggeration, 
        temperature, 
        seed_num, 
        cfgw
    )
    return (sr, wav)

def generate_raw(model, text, audio_prompt_path, exaggeration, temperature, seed_num, cfgw):
    if model is None:
        model = ChatterboxTTS.from_pretrained(DEVICE)

    if seed_num != 0:
        set_seed(int(seed_num))

    wav = model.generate(
        text,
        audio_prompt_path=audio_prompt_path,
        exaggeration=exaggeration,
        temperature=temperature,
        cfg_weight=cfgw,
    )
    return model.sr, wav.squeeze(0).numpy()


@app.route('/api/generate', methods=['POST'])
def api_generate():
    try:
        text = request.form.get('text', "Hello there. How are you?")
        voice_data = request.files.get('voice_data')
        seed = int(request.form.get('seed', 0))
        exaggeration = float(request.form.get('exaggeration', 0.5))
        temperature = float(request.form.get('temperature', 0.8))
        cfgw = float(request.form.get('cfgw', 0.5))
        
        voice_file_path = None
        if voice_data and hasattr(voice_data, 'read'):
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                tmp.write(voice_data.read())
                voice_file_path = tmp.name

        # Call generate_raw to get both sample rate and numpy array
        sr, wav_numpy_array = generate_raw(
            None, # Pass None so generate_raw loads the model if not already loaded
            text,
            voice_file_path,
            exaggeration,
            temperature,
            seed,
            cfgw
        )
        
        # Convert the numpy array to WAV bytes
        byte_io = io.BytesIO()
        wavfile.write(byte_io, sr, wav_numpy_array)
        byte_io.seek(0) # Rewind the stream to the beginning
        wav_data_bytes = byte_io.read() # Read the bytes
        
        return Response(
            wav_data_bytes, # Now this is bytes
            mimetype='audio/wav',
            headers={
                'Content-Disposition': 'attachment; filename=generated_audio.wav'
            }
        )
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

with gr.Blocks() as demo:
    model_state = gr.State(None)  # Loaded once per session/user

    with gr.Row():
        with gr.Column():
            text = gr.Textbox(
                value="Now let's make my mum's favourite. So three mars bars into the pan. Then we add the tuna and just stir for a bit, just let the chocolate and fish infuse. A sprinkle of olive oil and some tomato ketchup. Now smell that. Oh boy this is going to be incredible.",
                label="Text to synthesize (max chars 300)",
                max_lines=5
            )
            ref_wav = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Reference Audio File", value=None)
            exaggeration = gr.Slider(0.25, 2, step=.05, label="Exaggeration (Neutral = 0.5, extreme values can be unstable)", value=.5)
            cfg_weight = gr.Slider(0.0, 1, step=.05, label="CFG/Pace", value=0.5)

            with gr.Accordion("More options", open=False):
                seed_num = gr.Number(value=0, label="Random seed (0 for random)")
                temp = gr.Slider(0.05, 5, step=.05, label="temperature", value=.8)

            run_btn = gr.Button("Generate", variant="primary")

        with gr.Column():
            audio_output = gr.Audio(label="Output Audio")

    demo.load(fn=load_model, inputs=[], outputs=model_state)

    run_btn.click(
        fn=generate,
        inputs=[
            model_state,
            text,
            ref_wav,
            exaggeration,
            temp,
            seed_num,
            cfg_weight,
        ],
        outputs=audio_output,
    )

if __name__ == "__main__":
    flask_thread = threading.Thread(target=app.run, kwargs={
        "host": "0.0.0.0",
        "port": 7861
    })
    flask_thread.daemon = True
    flask_thread.start()

    demo.queue(
        max_size=50,
        default_concurrency_limit=1,
    ).launch(share=False, server_name="0.0.0.0", server_port=7860)