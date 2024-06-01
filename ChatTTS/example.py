from ChatTTS import Chat
import numpy as np
import wave


def main():
    chat = Chat()
    chat.load_models(source="local", local_path=r"D:\Python\TTSProject\ChatTTS\ChatTTS\models")

    r = chat.sample_random_speaker(seed=1112)
    params_infer_code = {
        "spk_emb": r,  # add sampled speaker
        "temperature": 0.3,  # using custom temperature
        "top_P": 0.7,  # top P decode
        "top_K": 20,  # top K decode
    }
    texts = "我常以人就这么一辈子这句话告诫本人并劝告友人。这七个字，说来轻易，听来简略，想起来却很深厚。它能使我在脆弱时变得英勇，自豪时变得谦逊，颓废时变得积极，苦楚时变得欢愉时，对任何事拿得起也放得下，所以我称它为当头棒喝、七字规语。——我常想世间的劳苦愁烦、恩恩怨怨，如有不能化解的，不能消受的，不也就过这短短的多少十年就云消雾散了吗？若是如此，又有什么解不开的呢？"
    wavs = chat.infer(texts, use_decoder=True, params_infer_code=params_infer_code)

    audio_data = np.array(wavs[0], dtype=np.float32)
    sample_rate = 2400
    audio_data = (audio_data * 32767).astype(np.int16)

    with wave.open("test2.wav", "w") as wf:
        wf.setnchannels(1)  # Mono channel
        wf.setsampwidth(2)  # 2 bytes per sample
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())


main()
