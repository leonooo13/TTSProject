import wave

import numpy as np

from ChatTTS.core import Chat


def text_to_speech(text, out_file, seed=123):
    if not text:
        raise ValueError("text is empty")

    chat = Chat()
    try:
        chat.load_models(source="local", local_path="ChatTTS/ChatTTS/models")
    except Exception as e:
        # this is a tricky for most newbies do not now the args for cli
        print("The model maybe broke will load again")
        chat.load_models(force_redownload=True)
    texts = [
        text,
    ]
    if seed:
        r = chat.sample_random_speaker(seed=seed)
        params_infer_code = {
            "spk_emb": r,  # add sampled speaker
            "temperature": 0.3,  # using custom temperature
            "top_P": 0.7,  # top P decode
            "top_K": 20,  # top K decode
        }
        wavs = chat.infer(texts, use_decoder=True, params_infer_code=params_infer_code)
    else:
        wavs = chat.infer(texts, use_decoder=True)

    audio_data = np.array(wavs[0], dtype=np.float32)
    sample_rate = 24000
    audio_data = (audio_data * 32767).astype(np.int16)

    with wave.open(out_file, "w") as wf:
        wf.setnchannels(1)  # Mono channel
        wf.setsampwidth(2)  # 2 bytes per sample
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())


if __name__ == '__main__':
    out_file = "demo1.wav"
    text = """大家好，今天我们来聊聊一个非常热门的话题——人工智能，简称AI。人工智能，听起来像是科幻小说里的概念，但其实它已经悄然走进了我们的生活。

AI是计算机科学的一个分支，它试图理解智能的实质，并生产出一种新的能以人类智能相似方式做出反应的智能机器。"""

    text_to_speech(text, out_file)
    out_file2 = "demo2.wav"
    text = """但人工智能并不只是冰冷的机器，它还能帮助我们解决很多实际问题。比如，AI可以帮助医生更准确地诊断疾病，帮助农民更有效地种植作物，甚至可以帮助我们找到丢失的宠物。它让我们的生活变得更加便捷和美好。"""
    text_to_speech(text, out_file2)
