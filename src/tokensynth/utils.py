import torch
import numpy as np
import pretty_midi
import itertools 
from pathlib import Path
import os
from huggingface_hub import hf_hub_download
import appdirs

def get_tokens_from_midi(file_path, pitch_shift=0):
    # Extract tokens from a MIDI file, with an optional pitch shift.
    # Each note is converted into four tokens:
    # 1) Start time  2) End time  3) Pitch (with shift)  4) Quantized velocity

    def quantize(value):
        # Return the index of the quantized value that is closest to the given velocity.
        quantized_values = [25, 50, 100, 127]
        diff = np.array([np.abs(q - value) for q in quantized_values])
        return diff.argmin()
        
    midi_file = pretty_midi.PrettyMIDI(file_path)
    notes = midi_file.instruments[0].notes
    tokens = []
    # Gather all notes from every instrument and then sort by start time.
    notes_list = [instruments.notes for instruments in midi_file.instruments]
    notes = list(itertools.chain(*notes_list))
    notes.sort(key=lambda x: x.start)

    # Only process notes that start or end before 5.0 seconds.
    for note in notes:
        if note.start >= 5.0 or note.end >= 5.0:
            break
        tokens.append(1 + int(note.start * 100))      # Encoding the start time
        tokens.append(1 + 500 + int(note.end * 100))  # Encoding the end time
        tokens.append(1 + 500 + 500 + note.pitch + pitch_shift)            # Encoding the pitch (with shift)
        tokens.append(1 + 500 + 500 + 128 + quantize(note.velocity))       # Encoding the quantized velocity
    
    return torch.tensor(tokens).reshape(1, -1)

def sample(logits, top_p=None, top_k=None, midi_vocab_size=None, audio_vocab_size=None, temperature=1.0):
    # Choose either top-p sampling or top-k sampling based on the provided arguments.
    # If top_p is not None, use nucleus sampling (top-p).
    # If top_k is not None, use top-k sampling.
    # Raises an error if neither is specified.

    def top_k_sampling(logits, k, midi_vocab_size, audio_vocab_size, temperature=1.0):
        # logits: shape (1, seq_len, midi_vocab_size + 9 * audio_vocab_size)
        # k: integer specifying top-k
        # Returns sampled indices for one time step.
        # This function applies top-k filtering and samples from the truncated probability distribution.

        sampled_indices = []
        midi_logits = logits[:, :midi_vocab_size]
        # Sort the MIDI logits in descending order, then zero out anything below the top-k.
        midi_sorted_logits, midi_indices = torch.sort(midi_logits, descending=True)
        midi_sorted_logits[:, k:midi_vocab_size] = -float("inf")
        midi_top_k_prob = torch.softmax(midi_sorted_logits, dim=-1)
        midi_sampled = torch.multinomial(midi_top_k_prob, 1)
        midi_reindex = torch.gather(midi_indices, dim=-1, index=midi_sampled)
        sampled_indices.append(midi_reindex)

        # Repeat the same top-k logic for each of the 9 audio token groups.
        for i in range(9):
            audio_logits = logits[:, midi_vocab_size + i*audio_vocab_size : midi_vocab_size + (i+1)*audio_vocab_size]
            audio_sorted_logits, audio_indices = torch.sort(audio_logits, descending=True)
            audio_sorted_logits[:, k+1:audio_vocab_size] = -float("inf")
            audio_top_k_prob = torch.softmax(audio_sorted_logits / temperature, dim=-1)
            audio_sampled = torch.multinomial(audio_top_k_prob, 1)
            audio_reindex = torch.gather(audio_indices, dim=-1, index=audio_sampled)
            sampled_indices.append(audio_reindex)
        
        # Concatenate all sampled indices and return.
        sampled_indices = torch.cat(sampled_indices, dim=-1).unsqueeze(0)
        return sampled_indices

    def top_p_sampling(logits, p, midi_vocab_size, audio_vocab_size, temperature=1.0):
        # logits: shape (1, midi_vocab_size + 9 * audio_vocab_size)
        # p: float specifying cumulative probability threshold
        # Returns sampled indices for one time step.
        # This function applies nucleus sampling (top-p) and samples from the filtered distribution.

        sampled_indices = []
        midi_logits = logits[:, :midi_vocab_size]
        # Sort the MIDI logits in descending order, then find the cutoff based on cumulative probability.
        midi_sorted_logits, midi_indices = torch.sort(midi_logits, descending=True)
        midi_prob = torch.softmax(midi_sorted_logits, dim=-1)
        midi_cum_prob = torch.cumsum(midi_prob, dim=-1)
        midi_index = torch.where(midi_cum_prob >= p, torch.ones_like(midi_cum_prob), torch.zeros_like(midi_cum_prob)).argmax(dim=-1)
        midi_sorted_logits[0, midi_index+1:] = -float("inf")
        midi_top_p_prob = torch.softmax(midi_sorted_logits, dim=-1)
        midi_sampled = torch.multinomial(midi_top_p_prob, 1)
        midi_reindex = torch.gather(midi_indices, dim=-1, index=midi_sampled)
        sampled_indices.append(midi_reindex)

        # Repeat the same top-p logic for each of the 9 audio token groups.
        for i in range(9):
            audio_logits = logits[:, midi_vocab_size + i*audio_vocab_size : midi_vocab_size + (i+1)*audio_vocab_size]
            audio_sorted_logits, audio_indices = torch.sort(audio_logits, descending=True)
            audio_prob = torch.softmax(audio_sorted_logits, dim=-1)
            audio_cum_prob = torch.cumsum(audio_prob, dim=-1)
            audio_index = torch.where(audio_cum_prob > p, torch.ones_like(audio_cum_prob), torch.zeros_like(audio_cum_prob)).argmax(dim=-1)
            audio_sorted_logits[0, audio_index+1:] = -float("inf")
            audio_top_p_prob = torch.softmax(audio_sorted_logits / temperature, dim=-1)
            audio_sampled = torch.multinomial(audio_top_p_prob, 1)
            audio_reindex = torch.gather(audio_indices, dim=-1, index=audio_sampled)
            sampled_indices.append(audio_reindex)

        # Concatenate all sampled indices and return.
        sampled_indices = torch.cat(sampled_indices, dim=-1).unsqueeze(0)
        return sampled_indices

    if top_p is not None:
        return top_p_sampling(logits, top_p, midi_vocab_size, audio_vocab_size, temperature)
    elif top_k is not None:
        return top_k_sampling(logits, top_k, midi_vocab_size, audio_vocab_size, temperature)
    else:
        raise ValueError("Either top_p or top_k must be provided.")

def post_process_audio_tokens(tokens):
    # Perform a simple post-processing step on audio tokens.
    # This function adjusts the token values by subtracting 1, ensuring they remain non-negative.
    seq_len = tokens.shape[1] - 8
    for i in range(9):
        tokens[:, :seq_len, i] = torch.max(
            torch.zeros(1).long().to(tokens.device),
            tokens[:, i:i+seq_len, i] - 1
        )
    return tokens[:, :seq_len, :]

def download_model(model_name):
    """Download the specified model checkpoint if it doesn't exist locally."""

    # Define model filenames
    model_filenames = {
        "clap": "clap_music_audioset_epoch_15_esc_90.14.pt",
        "dac": "dac_weights_44khz_8kbps_0.0.1.pt",
        "token_synth": "token_synth.pt",
        "token_synth_aug": "token_synth_aug.pt",
        "token_synth_unconditional": "token_synth_unconditional.pt",
    }

    if model_name not in model_filenames:
        raise ValueError(f"Unknown model: {model_name}")

    filename = model_filenames[model_name]

    # Determine the cache directory (OS-specific)
    cache_dir = Path(appdirs.user_cache_dir("tokensynth"))
    ckpt_path = cache_dir / filename

    # Download the model if it's not found locally
    if not ckpt_path.exists():
        os.makedirs(cache_dir, exist_ok=True)
        print(f"{filename} not found locally. Downloading from Hugging Face...")
        hf_hub_download(repo_id="KyungsuKim/TokenSynth", filename=filename, local_dir=cache_dir)

    return ckpt_path
