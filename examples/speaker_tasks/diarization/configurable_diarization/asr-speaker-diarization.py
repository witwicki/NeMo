#!/usr/bin/env python

### Automatic Speech Recognition with Speaker Diarization

# arguments
import sys
if len(sys.argv) < 2:
    print(f"\nUsage: {sys.argv[0]} [/path/to/input.wav]\n")
    sys.exit(0)
    
audio_file_path = sys.argv[1]

REALIGN_WITH_LM = False
ARPA_MODEL_URL = "https://kaldi-asr.org/models/5/4gram_big.arpa.gz"

import gzip
import shutil

# imports for nemo ASR pipeline
import nemo.collections.asr as nemo_asr
import numpy as np
from IPython.display import Audio, display
import librosa
import os
import wget
import matplotlib.pyplot as plt

import nemo
import glob

import torch

import pprint
pp = pprint.PrettyPrinter(indent=4)

# setup data directory
import shutil
ROOT = os.getcwd()
data_dir = os.path.join(ROOT,'data_v4.1')
os.makedirs(data_dir, exist_ok=True)
audio_file_name = os.path.basename(audio_file_path)
data_name = os.path.splitext(audio_file_name)[0]
data_file_path = os.path.join(data_dir,audio_file_name)
if not os.path.exists(data_file_path):
    shutil.copy2(audio_file_path, data_file_path)
AUDIO_FILENAME = data_file_path

audio_file_list = glob.glob(f"{data_dir}/*.wav")
print("All data files: \n", audio_file_list)

# display in a notebook waveform in a notebook
signal, sample_rate = librosa.load(AUDIO_FILENAME, sr=None)
#display(Audio(signal,rate=sample_rate))

def display_waveform(signal,text='Audio',overlay_color=[]):
    fig,ax = plt.subplots(1,1)
    fig.set_figwidth(20)
    fig.set_figheight(2)
    plt.scatter(np.arange(len(signal)),signal,s=1,marker='o',c='k')
    if len(overlay_color):
        plt.scatter(np.arange(len(signal)),signal,s=1,marker='o',c=overlay_color)
    fig.suptitle(text, fontsize=16)
    plt.xlabel('time (secs)', fontsize=18)
    plt.ylabel('signal strength', fontsize=14);
    plt.axis([0,len(signal),-0.5,+0.5])
    time_axis,_ = plt.xticks();
    plt.xticks(time_axis[:-1],time_axis[:-1]/sample_rate);
    
COLORS="b g c m y".split()

def get_color(signal,speech_labels,sample_rate=16000):
    c=np.array(['k']*len(signal))
    for time_stamp in speech_labels:
        start,end,label=time_stamp.split()
        start,end = int(float(start)*16000),int(float(end)*16000),
        if label == "speech":
            code = 'red'
        else:
            code = COLORS[int(label.split('_')[-1])]
        c[start:end]=code
    
    return c
#display_waveform(signal)


# Parameter setting for ASR and diarization
#  starting with basic transcription using standard nemo ASR model (QuartzNet15x5Base-En)
from omegaconf import OmegaConf
import shutil
DOMAIN_TYPE = "meeting" # Can be meeting or telephonic based on domain type of the audio file
CONFIG_FILE_NAME = f"diar_infer_{DOMAIN_TYPE}.yaml"

CONFIG_URL = f"https://raw.githubusercontent.com/NVIDIA/NeMo/main/examples/speaker_tasks/diarization/conf/inference/{CONFIG_FILE_NAME}"

if not os.path.exists(os.path.join(data_dir,CONFIG_FILE_NAME)):
    CONFIG = wget.download(CONFIG_URL, data_dir)
else:
    CONFIG = os.path.join(data_dir,CONFIG_FILE_NAME)

cfg = OmegaConf.load(CONFIG)
cfg.num_workers = 1
cfg.batch_size = 1
print(OmegaConf.to_yaml(cfg))

# Create a manifest file for input with below format. 
# {"audio_filepath": "/path/to/audio_file", "offset": 0, "duration": null, "label": "infer", "text": "-", 
# "num_speakers": null, "rttm_filepath": "/path/to/rttm/file", "uem_filepath"="/path/to/uem/filepath"}
import json
meta = {
    'audio_filepath': AUDIO_FILENAME, 
    'offset': 0, 
    'duration':None, 
    'label': 'infer', 
    'text': '-', 
    'num_speakers': None, 
    'rttm_filepath': None, 
    'uem_filepath' : None
}
with open(os.path.join(data_dir,'input_manifest.json'),'w') as fp:
    json.dump(meta,fp)
    fp.write('\n')

cfg.diarizer.manifest_filepath = os.path.join(data_dir,'input_manifest.json')
#!cat {cfg.diarizer.manifest_filepath}

# obtain voice activity labels from ASR using Neural VAD and Conformer ASR 
pretrained_speaker_model = 'titanet_large' # 'ecapa_tdnn' 
cfg.diarizer.manifest_filepath = cfg.diarizer.manifest_filepath
cfg.diarizer.out_dir = data_dir #Directory to store intermediate files and prediction outputs
cfg.diarizer.speaker_embeddings.model_path = pretrained_speaker_model
cfg.diarizer.clustering.parameters.oracle_num_speakers=False
cfg.diarizer.vad.model_path = 'vad_multilingual_marblenet'
cfg.diarizer.asr.model_path = 'nvidia/parakeet-ctc-1.1b' #'stt_en_fastconformer_ctc_large' #'nvidia/parakeet-tdt_ctc-0.6b' #'stt_en_conformer_ctc_large'
cfg.diarizer.oracle_vad = False # ----> Not using oracle VAD 
cfg.diarizer.asr.parameters.asr_based_vad = False
cfg.diarizer.asr.parameters.asr_batch_size = 1

#print(f"cfg.diarizer.asr.parameters={cfg.diarizer.asr.parameters}")
#print(f"cfg.diarizer.asr.ctc_decoder_parameters={cfg.diarizer.asr.ctc_decoder_parameters}")
#input("<press any key to continue>")

# Run ASR and get word timestamps
print("Running ASR with word timestamping...")
#input("<press any key to continue>")
from nemo.collections.asr.parts.utils.decoder_timestamps_utils import ASRDecoderTimeStamps
asr_decoder_ts = ASRDecoderTimeStamps(cfg.diarizer)
print(asr_decoder_ts)
asr_model = asr_decoder_ts.set_asr_model()
torch.cuda.empty_cache()
#input("<press any key to continue>")
word_hyp, word_ts_hyp = asr_decoder_ts.run_ASR(asr_model)

print("Decoded word output dictionary: \n", word_hyp[data_name])
print("Word-level timestamps dictionary: \n", word_ts_hyp[data_name])
#input("<press any key to continue>")
# Match diarization results with ASR outputs
from nemo.collections.asr.parts.utils.diarization_utils import OfflineDiarWithASR
asr_diar_offline = OfflineDiarWithASR(cfg.diarizer)
asr_diar_offline.word_ts_anchor_offset = asr_decoder_ts.word_ts_anchor_offset

# Run diarization with the extracted word timestamps
diar_hyp, diar_score = asr_diar_offline.run_diarization(cfg, word_ts_hyp)
print("Diarization hypothesis output: \n", diar_hyp[data_name])

def read_file(path_to_file):
    with open(path_to_file) as f:
        contents = f.read().splitlines()
    return contents

predicted_speaker_label_rttm_path = f"{data_dir}/pred_rttms/{data_name}.rttm"
pred_rttm = read_file(predicted_speaker_label_rttm_path)

#pp.pprint(pred_rttm)

from nemo.collections.asr.parts.utils.speaker_utils import rttm_to_labels
pred_labels = rttm_to_labels(predicted_speaker_label_rttm_path)

color = get_color(signal, pred_labels)
#display_waveform(signal,'Audio with Speaker Labels', color)
#display(Audio(signal,rate=16000))

# Check the speaker-labeled ASR transcription output
trans_info_dict = asr_diar_offline.get_transcript_with_speaker_labels(diar_hyp, word_hyp, word_ts_hyp, repunctuate=True)
transcription_path_to_file = f"{data_dir}/pred_rttms/{data_name}.txt"
transcript = read_file(transcription_path_to_file)
#pp.pprint(transcript)
transcription_path_to_file = f"{data_dir}/pred_rttms/{data_name}.json"
json_contents = read_file(transcription_path_to_file)
#pp.pprint(json_contents)

# (Experimental) Realign words with Language model
#   Diarization result with ASR transcript can be enhanced by applying a language model.
#   The mapping between speaker labels and words can be realigned by employing language models.
#   The realigning process calculates the probability of the words around the boundary between
#   two hypothetical sentences spoken by different speakers.
if REALIGN_WITH_LM:
    arpa_model_path = os.path.join(data_dir, 'models', '4gram_big.arpa')
    arpa_model_directory = os.path.dirname(arpa_model_path)
    if not os.path.exists(arpa_model_path):
        os.makedirs(arpa_model_directory, exist_ok=True)
        print(f"Downloading and unzipping ARPA model to {arpa_model_path} ...")
        wget.download(ARPA_MODEL_URL, arpa_model_directory)
        # unzip
        with gzip.open(os.path.join(data_dir, 'models', '4gram_big.arpa.gz'), 'rb') as f_in:
            with open(arpa_model_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

    cfg.diarizer.asr.realigning_lm_parameters.arpa_language_model = arpa_model_path
    cfg.diarizer.asr.realigning_lm_parameters.logprob_diff_threshold = 1.2

    import importlib
    import nemo.collections.asr.parts.utils.diarization_utils as diarization_utils
    importlib.reload(diarization_utils) # This module should be reloaded after you install arpa.

    # Create a new instance with realigning language model
    asr_diar_offline = OfflineDiarWithASR(cfg.diarizer)
    asr_diar_offline.word_ts_anchor_offset = asr_decoder_ts.word_ts_anchor_offset

    asr_diar_offline.get_transcript_with_speaker_labels(diar_hyp, word_hyp, word_ts_hyp)

    transcription_path_to_file = f"{data_dir}/pred_rttms/{data_name}.txt"
    transcript = read_file(transcription_path_to_file)
    pp.pprint(transcript)

    

