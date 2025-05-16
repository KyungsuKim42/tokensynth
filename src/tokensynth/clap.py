import numpy as np
import laion_clap
from pathlib import Path
import librosa
import torch
import torch.nn.functional as F
import warnings
from transformers import logging
from huggingface_hub import hf_hub_download

from tokensynth import utils

class CLAP:
    """
    A class to handle audio and text embeddings using the CLAP model.
    
    Attributes:
        device (torch.device): The device (CPU or GPU) on which the model and tensors will be placed.
        clap (laion_clap.CLAP_Module): Instance of the LAION CLAP model.
    """
    def __init__(self, device=None):
        """
        Initialize the CLAP class.
        Automatically selects GPU if available, otherwise CPU.
        Initializes the LAION CLAP model, loads the checkpoint, and sets the model to evaluation mode.
        
        Args:
            device (torch.device, optional): The desired device for the model. Defaults to None for auto-selection.
        """
        # Suppress non-critical warnings
        warnings.filterwarnings('ignore', message='torch.meshgrid:.*')
        # Disable transformers library's warnings
        logging.set_verbosity_error()
        
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize the CLAP model
        self.clap = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base')

        # Download and load the CLAP checkpoint
        ckpt_path = utils.download_model('clap')
        self.clap.load_ckpt(ckpt_path, verbose=False)       
        self.clap.to(self.device)
        
        # Set model to evaluation mode
        self.clap.eval()

    def encode_audio(self, audio):
        """
        Load an audio file, downsample to 16kHz, then upsample to 48kHz and retrieve the CLAP audio embedding.
        
        Args:
            audio (str, torch.Tensor, np.ndarray): Audio data.
        
        Returns:
            torch.Tensor: A tensor containing the audio embedding.
        """
        # Load audio at 16kHz, then resample to 48kHz
        if isinstance(audio, str):
            audio, sr = librosa.load(audio, sr=16000)
        elif isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()
        elif isinstance(audio, np.ndarray):
            audio = audio
        else:
            raise ValueError("audio must be a string or a torch.Tensor")

        audio = librosa.resample(audio, orig_sr=16000, target_sr=48000)
        
        # Convert audio data to a tensor and move it to the device
        audio = torch.tensor(audio.reshape(1,-1)).float().to(self.device)
        
        # Get audio embedding from the CLAP model
        return self.clap.get_audio_embedding_from_data(audio, use_tensor=True)
    
    def encode_text(self, text):
        """
        Retrieve CLAP text embedding from a single string input.
        
        Args:
            text (str): A string containing the input text.
        
        Returns:
            torch.Tensor: A tensor containing the text embedding.
        
        Raises:
            AssertionError: If the provided text is not a string.
        """
        assert isinstance(text, str), "text must be a string"
        return self.clap.get_text_embedding([text], use_tensor=True)