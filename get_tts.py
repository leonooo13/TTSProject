###################################
# Sample a speaker from Gaussian.
import torch
std, mean = torch.load('ChatTTS/asset/spk_stat.pt').chunk(2)
rand_spk = torch.randn(768) * std + mean
chat = ChatTTS.Chat()
chat.load_models()
params_infer_code = {
  'spk_emb': rand_spk, # add sampled speaker
  'temperature': .3, # using custom temperature
  'top_P': 0.7, # top P decode
  'top_K': 20, # top K decode
}

###################################
# For sentence level manual control.

# use oral_(0-9), laugh_(0-2), break_(0-7)
# to generate special token in text to synthesize.
params_refine_text = {
  'prompt': '[oral_2][laugh_0][break_6]'
}

wav = chat.infer("<PUT YOUR TEXT HERE>", params_refine_text=params_refine_text, params_infer_code=params_infer_code)

###################################
# For word level manual control.
text = 'What is [uv_break]your favorite english food?[laugh][lbreak]'
wav = chat.infer(text, skip_refine_text=True, params_infer_code=params_infer_code)
