import pytest
import torch
from tokensynth.dac_decoder import DACDecoder
from tokensynth.clap import CLAP
from tokensynth.model import TokenSynth
import numpy as np
# -- fixtures -------------------------------------------------------------------------------------

@pytest.fixture(scope='module')
def device():
    # Determine the device to use (CUDA if available, otherwise CPU)
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@pytest.fixture(scope='module')
def dac_decoder(device):
    # Initialize DACDecoder with the specified device
    return DACDecoder(device=device)

@pytest.fixture(scope='module')
def clap(device):
    # Initialize CLAP with the specified device
    return CLAP(device=device)

@pytest.fixture(scope='module')
def token_synth(device):
    # Initialize TokenSynth with the specified device
    return TokenSynth.from_pretrained(aug=True, device=device)

# -- tests ----------------------------------------------------------------------------------------

def test_dac_decoder_initialization(dac_decoder):
    # Test if DACDecoder is initialized correctly
    assert dac_decoder.dac_model is not None
    assert dac_decoder.resampler is not None

@pytest.mark.parametrize('audio_tokens', [torch.randint(0, 1024, (1, 100, 9))])
def test_dac_decoder_decode(dac_decoder, device, audio_tokens):
    # Test the decode function of DACDecoder
    print(device)
    audio_tokens = audio_tokens.to(device)  # Apply device at this point
    audio_16k = dac_decoder.decode(audio_tokens)
    assert audio_16k is not None
    assert not torch.any(torch.isnan(audio_16k)), "Decoded audio should not contain NaN values"
    assert torch.all((audio_16k >= -1) & (audio_16k <= 1)), "Decoded audio should be between -1 and 1"

def test_clap_initialization(clap):
    # Test if CLAP is initialized correctly
    assert clap.clap is not None

@pytest.mark.parametrize('audio_fname', ["media/reference_audio.wav"])  # Replace with a valid audio file path
def test_clap_encode_audio(clap, audio_fname):
    # Test the encode_audio function of CLAP
    audio_embedding = clap.encode_audio(audio_fname)
    assert audio_embedding is not None
    assert not torch.any(torch.isnan(audio_embedding)), "Audio embedding should not contain NaN values"
    assert audio_embedding.shape[1] == 512  # Assuming embedding size is 512

@pytest.mark.parametrize('text', ["This is a test text."])
def test_clap_encode_text(clap, text):
    # Test the encode_text function of CLAP
    text_embedding = clap.encode_text(text)
    assert text_embedding is not None
    assert not torch.any(torch.isnan(text_embedding)), "Text embedding should not contain NaN values"
    assert text_embedding.shape[1] == 512  # Assuming embedding size is 512

def test_token_synth_initialization(token_synth):
    # Test if TokenSynth is initialized correctly
    assert token_synth is not None

@pytest.mark.parametrize('midi_fname', [("media/input_midi.mid")])
def test_token_synth_generate(token_synth, midi_fname, device):
    # Test the synthesize function of TokenSynth
    clap_embedding = torch.randn(1, 512).to(device)
    clap_embedding = clap_embedding / torch.norm(clap_embedding, p=2)
    output = token_synth.synthesize(clap_embedding, midi_fname, top_p=0.9)
    assert output is not None
    assert not torch.any(torch.isnan(output)), "Output should not contain NaN values"
    assert output.shape[2] == 9  # Assuming output size is 9 for audio tokens
    assert torch.all((output >= 0) & (output <= 1023)), "Output integers should be between 0 and 1023"
