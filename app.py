import gradio as gr
from TTS.api import TTS

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v1")
tts.to("cuda")


def predict(prompt, language, audio_file_pth, audio_mic, agree):

    if audio_mic is not None:
        audio = audio_mic
    else:
        audio = audio_file_pth
        
    if agree == True:
        tts.tts_to_file(
            text=prompt,
            file_path="output.wav",
            speaker_wav=audio,
            language=language,
        )

        return (
            gr.make_waveform(
                audio="output.wav",
            ),
            "output.wav",
        )
    else:
        gr.Warning("Please accept the Terms & Condition!")


title = "Coqui🐸 XTTS"

description = """
<a href="https://huggingface.co/coqui/XTTS-v1">XTTS</a> is a Voice generation model that lets you clone voices into different languages by using just a quick 3-second audio clip. 
<br/>
Built on Tortoise, XTTS has important model changes that make cross-language voice cloning and multi-lingual speech generation super easy. 
<br/>
This is the same model that powers Coqui Studio, and Coqui API, however we apply a few tricks to make it faster and support streaming inference.
<br/>
<br/>
<p>For faster inference without waiting in the queue, you should duplicate this space and upgrade to GPU via the settings.
<br/>
<a href="https://huggingface.co/spaces/coqui/xtts?duplicate=true">
<img style="margin-top: 0em; margin-bottom: 0em" src="https://bit.ly/3gLdBN6" alt="Duplicate Space"></a>
</p>
"""

article = """
<div style='margin:20px auto;'>
<p>By using this demo you agree to the terms of the Coqui Public Model License at https://coqui.ai/cpml</p>
</div>
"""

gr.Interface(
    fn=predict,
    inputs=[
        gr.Textbox(
            label="Text Prompt",
            info="One or two sentences at a time is better",
            lines=3,
            placeholder="想说却还没说的 还很多"
        ),
        gr.Dropdown(
            label="Language",
            info="Select an output language for the synthesised speech",
            choices=[
                "en",
                "es",
                "fr",
                "de",
                "it",
                "pt",
                "pl",
                "tr",
                "ru",
                "nl",
                "cz",
                "ar",
                "zh",
            ],
            max_choices=1,
            value="zh-cn",
        ),
        gr.Audio(
            label="通过文件上传语音",
            type="filepath",
        ),
        gr.Audio(
            label="使用麦克风上传语音",
            type="filepath",
            source="microphone",
        ),
        gr.Checkbox(
            label="使用条款",
            value=True,
            info="我承诺：不会利用此程序生成对个人或组织造成侵害的任何内容",
        ),
    ],
    outputs=[
        gr.Video(label="为您合成的专属音频"),
        gr.Audio(label="Synthesised Audio", visible=False),
    ],
    title=title,
    description=description,
    article=article,
).queue().launch(share=True, debug=True)
