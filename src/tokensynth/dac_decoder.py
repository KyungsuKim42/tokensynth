import dac  # Descript Audio Codec
import torch
import torchaudio
import warnings
from pathlib import Path

from tokensynth import utils

class DACDecoder:
    """
    DACDecoder class for decoding audio tokens to waveforms.

    This class utilizes a Descript Audio Codec (DAC) model to decode
    audio tokens into waveforms, then resamples the output to 16kHz.

    Args:
        device (torch.device, optional): The computation device (CPU or GPU).
            Defaults to the available GPU if available, otherwise CPU.
    """

    def __init__(self, device=None):
        """
        Initialize the DACDecoder.

        Suppresses certain deprecation warnings, downloads the DAC model
        checkpoint if not found locally, and sets up a resampler from 44.1kHz
        to 16kHz.

        Args:
            device (torch.device, optional): Target device for model and
                resampler. If None, defaults to GPU if available, else CPU.
        """
        # Suppress weight_norm deprecation warning
        warnings.filterwarnings('ignore', message='torch.nn.utils.weight_norm is deprecated.*')
     
        ckpt_path = utils.download_model('dac')
        checkpoint_path = Path(ckpt_path)  # Convert to pathlib Path

        self.dac_model = dac.DAC.load(checkpoint_path)
        self.dac_model.eval()
        self.dac_model.to(device)

        # Resampler from 44.1 kHz to 16 kHz
        self.resampler = torchaudio.transforms.Resample(orig_freq=44100, new_freq=16000)
        self.resampler.to(device)
    
    def decode(self, audio_tokens):
        """
        Decode audio tokens into a 16kHz waveform.

        Permutes the input tokens to match the expected format for the
        quantizer, reconstructs the latent representation, decodes it
        with the DAC model, and resamples the result to 16kHz.

        Args:
            audio_tokens (torch.Tensor): A tensor of audio tokens with shape
                (batch_size, 100, 1024) or similar dimensions. The last two
                dimensions are permuted internally.

        Returns:
            torch.Tensor: The decoded single-channel audio waveform at
                16kHz sample rate.
        """
        audio_tokens = audio_tokens.permute(0, 2, 1)
        latent = self.dac_model.quantizer.from_codes(audio_tokens)[0]
        audio_44k = self.dac_model.decode(latent)
        audio_16k = self.resampler(audio_44k).squeeze(0).cpu()
        return audio_16k

if __name__ == "__main__":
    # Example usage of DACDecoder
    dac_decoder = DACDecoder()
    audio_tokens = torch.randn(1, 100, 1024)
    audio_16k = dac_decoder.decode(audio_tokens)
    print(audio_16k.shape)
