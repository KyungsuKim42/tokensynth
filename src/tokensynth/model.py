import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
from pathlib import Path
from huggingface_hub import hf_hub_download

from tokensynth import utils

class TokenSynth(nn.Module):
    """Transformer-based model for conditional audio synthesis from MIDI and CLAP embeddings.
    
    Attributes:
        device (torch.device): Computation device (CPU/GPU)
        hparams (HyperParameters): Model configuration parameters
        midi_tok_embed (nn.Embedding): Embedding layer for MIDI tokens
        audio_tok_embed (nn.ModuleList): Embedding layers for audio tokens (9 layers)
        pos_embed (nn.Parameter): Learnable positional embeddings
        clap_projection (nn.Sequential): Projection network for CLAP embeddings
        blocks (nn.Sequential): Stack of transformer blocks
        current_length (int): Tracks sequence position during generation
    """
    
    def __init__(self, device=None, load_unconditional_model=False):
        """Initialize TokenSynth model.
        
        Args:
            device (torch.device, optional): Target computation device. Auto-detected if None.
            load_unconditional_model (bool): Whether to load unconditional model for guidance
        """
        super().__init__()
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        hparams = self.get_hparams()
        self.hparams = hparams
        
        # Embedding layers
        self.midi_tok_embed = nn.Embedding(hparams.midi_vocab_size, hparams.embed_dim)
        self.audio_tok_embed = nn.ModuleList([
            nn.Embedding(hparams.audio_vocab_size, hparams.embed_dim) for _ in range(9)
        ])
        self.pos_embed = nn.Parameter(torch.zeros(1, hparams.max_len, hparams.embed_dim))
        self.clap_projection = nn.Sequential(
            nn.Linear(512, 1024), 
            nn.ReLU(), 
            nn.Linear(1024, hparams.embed_dim)
        )

        # Transformer architecture components
        self.dropout = nn.Dropout(hparams.embed_dropout)
        self.blocks = nn.Sequential(*[Block(hparams) for _ in range(hparams.num_blocks)])
        self.ln = nn.LayerNorm(hparams.embed_dim)
        self.fc = nn.Linear(hparams.embed_dim, hparams.midi_vocab_size + 9 * hparams.audio_vocab_size)

        self.current_length = 0  # Tracks generation position
        self.to(self.device)

        if load_unconditional_model:
            self.load_unconditional_model()

    def get_hparams(self):
        """Define model hyperparameters.
        
        Returns:
            HyperParameters: Nested class containing model configuration
        """
        class HyperParameters:
            def __init__(self):
                # Architecture configuration
                self.batch_size = 1
                self.num_heads = 16
                self.num_blocks = 12
                self.embed_dim = 1024
                
                # Regularization
                self.attn_dropout = 0.1
                self.embed_dropout = 0.1
                self.ff_dropout = 0.1
                
                # Vocabulary sizes
                self.midi_vocab_size = 1000 + 128 + 4 + 1  # [range, pitch, duration, special]
                self.audio_vocab_size = 1024 + 1  # Audio tokens + padding
                self.max_len = 1024  # Maximum sequence length
        return HyperParameters()
    
    @classmethod
    def from_pretrained(cls, path=None, aug=True, device=None):
        """Load pretrained model from checkpoint.
        
        Args:
            path (str, optional): Local path to checkpoint file
            aug (bool): Load data-augmented version if True
            device (torch.device, optional): Target computation device
            
        Returns:
            TokenSynth: Initialized model with loaded weights
        """
        model = cls(device=device)

        if path:
            checkpoint_path = Path(path)
        else:
            model_name = 'token_synth_aug' if aug else 'token_synth'
            checkpoint_path = utils.download_model(model_name)

        print(f"Loading checkpoint from {checkpoint_path}...")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)
        return model
    
    def synthesize(self, clap_embedding, midi_fname, context_tokens=None, top_p=None, top_k=None, guidance_scale=None, return_context_tokens=False, temperature=1.0, fnt=False):
        """Generate audio tokens from MIDI input with optional guidance.
        
        Args:
            clap_embedding (torch.Tensor): CLAP audio embedding [batch_size, 512]
            midi_fname (str): Path to MIDI file
            top_p (float, optional): Nucleus sampling probability
            top_k (int, optional): Top-k sampling cutoff
            guidance_scale (float, optional): Strength of unconditional guidance
            return_context_tokens (bool): Whether to return context tokens together with generated tokens
            temperature (float): Temperature for sampling
            fnt (bool): Whether to use first note temperature
        Returns:
            torch.Tensor: Generated audio tokens [batch_size, seq_len, 9]
            
        Raises:
            AssertionError: If required arguments are missing
        """
        # Input validation
        assert top_p or top_k, "Must provide either top_p or top_k for sampling"
        input_temp = temperature
        
        if guidance_scale and not hasattr(self, 'unconditional_model'):
            print("Loading unconditional model for guidance...")
            self.load_unconditional_model()

        self.eval()  # Set evaluation mode
        
        # Process MIDI input
        midi_tokens = utils.get_tokens_from_midi(midi_fname).to(self.device)
        midi_len = midi_tokens.shape[1]
        
        # Initialize token tensor [batch, seq_len, 10 (1 midi + 9 audio)]
        tokens = torch.zeros((1, self.hparams.max_len, 10), device=self.device).long()
        tokens[0, :midi_len, 0] = midi_tokens[0]

        if context_tokens is not None:
            tokens[0, midi_len+1:midi_len+1+context_tokens.shape[1], 1:] = context_tokens
            prefix_len = midi_len+1+context_tokens.shape[1]
        else:
            prefix_len = midi_len+1

        # Generation loop   
        self.count = 0  # Tracks guided steps
        self.fnt_count = 0
        logit = self.forward(tokens[:, :prefix_len], clap_embedding, use_cache=False)[:, -1, :]

        max_gen_length = min(prefix_len+451, self.hparams.max_len-1)  # Prevent overflow
        for i in tqdm(range(prefix_len, max_gen_length), desc="Synthesizing", leave=False):
            # Apply first note guidance
            if guidance_scale:
                is_silence = logit[:, self.hparams.midi_vocab_size:self.hparams.midi_vocab_size+1024].argmax().item() == 569
                if not is_silence and self.count < 9:
                    self.count += 1
                    logit = self.apply_first_note_guidance(logit, guidance_scale, midi_len, i, tokens)
            if fnt:
                is_silence = logit[:, self.hparams.midi_vocab_size:self.hparams.midi_vocab_size+1024].argmax().item() == 569
                if not is_silence and self.fnt_count < 9:
                    self.fnt_count += 1
                    temperature = input_temp
                else:
                    temperature = 1.0
            # Sample next token
            next_token = utils.sample(
                logit, 
                top_p=top_p,
                top_k=top_k,
                midi_vocab_size=self.hparams.midi_vocab_size,
                audio_vocab_size=self.hparams.audio_vocab_size,
                temperature=temperature
            )

            if (next_token == 0).all():  # End token
                break
                
            tokens[0, i, :] = next_token
            logit = self.forward(tokens[:, i:i+1], use_cache=True)[:, -1, :]

        # Process output tokens
        if return_context_tokens:   
            audio_tokens = tokens[:, midi_len+1:i, 1:]
        else:
            audio_tokens = tokens[:, prefix_len+1:i, 1:]
        return utils.post_process_audio_tokens(audio_tokens)
    
    def forward(self, x, clap_embedding=None, use_cache=False):
        """Transformer forward pass with optional CLAP conditioning.
        
        Args:
            x (torch.Tensor): Input tokens [batch_size, seq_len, 10]
            clap_embedding (torch.Tensor, optional): CLAP embedding [batch_size, 512]
            use_cache (bool): Enable KV caching for generation
            
        Returns:
            torch.Tensor: Output logits [batch_size, seq_len, vocab_size]
            
        Raises:
            AssertionError: If sequence exceeds max length
        """
        # Embed MIDI and audio tokens
        tok_embedding = self.midi_tok_embed(x[:,:,0])
        for i in range(9):
            tok_embedding += self.audio_tok_embed[i](x[:,:,i+1])
        
        # Add CLAP conditioning
        if clap_embedding is not None:
            assert not use_cache, "CLAP embedding only used in initial pass"
            clap_proj = self.clap_projection(clap_embedding).unsqueeze(1)
            embedding = torch.cat((clap_proj, tok_embedding), dim=1)
        else:
            embedding = tok_embedding
            
        # Positional embeddings
        seq_len = embedding.size(1)
        assert self.current_length + seq_len <= self.hparams.max_len, "Sequence exceeds max length"
        
        if use_cache:
            embedding += self.pos_embed[:, self.current_length:self.current_length+seq_len, :]
            self.current_length += seq_len
        else:
            embedding += self.pos_embed[:, :seq_len, :]
            self.current_length = seq_len
        
        # Transformer processing
        x = self.dropout(embedding)
        for block in self.blocks:
            x = block(x, use_cache=use_cache)
        return self.fc(self.ln(x))
    
    def tokens_to_audio(self, audio_tokens):
        """Convert audio tokens to waveform using DAC decoder.
        
        Args:
            audio_tokens (torch.Tensor): Generated audio tokens [batch_size, seq_len, 9]
            
        Returns:
            torch.Tensor: Audio waveform [samples]
        """
        audio_tokens = audio_tokens.permute(0, 2, 1)
        latent = self.dac_model.quantizer.from_codes(audio_tokens)[0]
        return self.dac_model.decode(latent)['audio'][0]
    
    def load_unconditional_model(self):
        """Load unconditional model for classifier-free guidance."""
        self.unconditional_model = TokenSynthUnconditional(device=self.device)  # Add device argument
        path = utils.download_model('token_synth_unconditional')
        self.unconditional_model.load_state_dict(torch.load(path, map_location=self.device), strict=False)
        self.unconditional_model.eval().to(self.device)
        print("Unconditional model loaded.")
    
    def apply_first_note_guidance(self, logit, guidance_scale, midi_len, i, tokens):
        """Blend conditional and unconditional logits for guidance.
        
        Args:
            logit (torch.Tensor): Current conditional logits
            guidance_scale (float): Guidance strength (0-1)
            midi_len (int): Length of MIDI input
            i (int): Current generation step
            tokens (torch.Tensor): All generated tokens
            
        Returns:
            torch.Tensor: Blended logits
        """
        midi_vocab = self.hparams.midi_vocab_size
        audio_vocab = self.hparams.audio_vocab_size
        
        if self.count == 1:  # First guided step
            # Add check for valid context length
            if i > midi_len+1:
                ctx = tokens[:, midi_len+1:i]
                uncond_logit = self.unconditional_model(ctx, use_cache=False)[:, -1, :]
                logit[:, midi_vocab:midi_vocab+audio_vocab*2] = \
                    guidance_scale * logit[:, midi_vocab:midi_vocab+audio_vocab*2] + \
                    (1-guidance_scale) * uncond_logit[:, midi_vocab:midi_vocab+audio_vocab*2]
        else:  # Subsequent steps
            ctx = tokens[:, -2:-1]  # Use cache
            uncond_logit = self.unconditional_model(ctx, use_cache=True)[:, -1, :]
            logit[:, midi_vocab:midi_vocab+audio_vocab*self.count] = \
                guidance_scale * logit[:, midi_vocab:midi_vocab+audio_vocab*self.count] + \
                (1-guidance_scale) * uncond_logit[:, midi_vocab:midi_vocab+audio_vocab*self.count]
        return logit


class TokenSynthUnconditional(TokenSynth):
    """Unconditional variant with start-of-sequence token.
    
    Attributes:
        sos_tok_embed (nn.Embedding): Start-of-sequence token embedding
    """
    
    def __init__(self, device=None):
        """Initialize unconditional model."""
        super().__init__()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sos_tok_embed = nn.Embedding(1, self.hparams.embed_dim)
        self.to(self.device)

    @classmethod
    def from_pretrained(cls, path=None, device=None):
        """Load pretrained model from checkpoint.
        
        Args:
            path (str, optional): Local path to checkpoint file
            device (torch.device, optional): Target computation device
            
        Returns:
            TokenSynth: Initialized model with loaded weights
        """
        model = cls(device=device)

        if path:
            checkpoint_path = Path(path)
        else:
            checkpoint_path = utils.download_model('token_synth_unconditional')

        print(f"Loading checkpoint from {checkpoint_path}...")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)
        return model
    
    def synthesize(self, x, use_cache=False):
        """ Autoregressively generate audio tokens from input tokens.
        
        Args:
            x (torch.Tensor): Input tokens [batch_size, seq_len, 10], Delay pattern should be applied.
            use_cache (bool): Enable KV caching
        """
        self.eval()
        tokens = torch.zeros((1, self.hparams.max_len, 10), device=self.device).long()
        tokens[0, :x.shape[1], 1:] = x

        i = x.shape[1]
        logit = self.forward(tokens[:, :i], use_cache=False)[:, -1, :]
        max_steps = self.hparams.max_len - i
        for _ in tqdm(range(max_steps), desc="Synthesizing"):
            next_token = utils.sample(logit, top_p=0.95, midi_vocab_size=self.hparams.midi_vocab_size, audio_vocab_size=self.hparams.audio_vocab_size)
            if (next_token == 0).all():  # End token
                break
            logit = self.forward(next_token, use_cache=True)[:, -1, :]
            tokens[0, i, :] = next_token
            i += 1
            
        # Process output tokens
        audio_tokens = tokens[:, :i, 1:]
        return utils.post_process_audio_tokens(audio_tokens)


    def forward(self, x, use_cache=False):
        """Forward pass with SOS token for unconditional generation.
        
        Args:
            x (torch.Tensor): Input tokens [batch_size, seq_len, 10]
            use_cache (bool): Enable KV caching
            
        Returns:
            torch.Tensor: Output logits [batch_size, seq_len, vocab_size]
        """
        # Embed tokens
        tok_embedding = self.midi_tok_embed(x[:,:,0])
        for i in range(9):
            tok_embedding += self.audio_tok_embed[i](x[:,:,i+1])
        
        # Add SOS token
        if not use_cache:
            sos = self.sos_tok_embed(torch.zeros(x.size(0), 1, device=x.device).long())
            embedding = torch.cat((sos, tok_embedding), dim=1)
        else:
            embedding = tok_embedding
            
        # Positional embeddings
        seq_len = embedding.size(1)
        assert self.current_length + seq_len <= self.hparams.max_len, "Sequence too long"
        
        if use_cache:
            embedding += self.pos_embed[:, self.current_length:self.current_length+seq_len, :]
            self.current_length += seq_len
        else:
            embedding += self.pos_embed[:, :seq_len, :] 
            self.current_length = seq_len
        
        # Transformer processing
        x = self.dropout(embedding)
        for block in self.blocks:
            x = block(x, use_cache=use_cache)
        return self.fc(self.ln(x))


class Block(nn.Module):
    """Transformer block with self-attention and feed-forward network.
    
    Attributes:
        ln1 (nn.LayerNorm): Pre-attention layer norm
        ln2 (nn.LayerNorm): Pre-FFN layer norm
        attn (MultiheadAttention): Attention module
        ff (nn.Sequential): Feed-forward network
    """
    
    def __init__(self, hparams):
        """Initialize transformer block.
        
        Args:
            hparams (HyperParameters): Model configuration
        """
        super().__init__()
        embed_dim = hparams.embed_dim
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.attn = MultiheadAttention(hparams)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(hparams.ff_dropout),
        )
    
    def forward(self, x, use_cache=False):
        """Block forward pass with residual connections.
        
        Args:
            x (torch.Tensor): Input tensor [batch_size, seq_len, embed_dim]
            use_cache (bool): Enable KV caching
            
        Returns:
            torch.Tensor: Output tensor [batch_size, seq_len, embed_dim]
        """
        x = x + self.attn(self.ln1(x), use_cache=use_cache)
        return x + self.ff(self.ln2(x))


class MultiheadAttention(nn.Module):
    """Multi-head attention with KV caching and causal masking.
    
    Attributes:
        num_heads (int): Number of attention heads
        head_dim (int): Dimension per attention head
        key (nn.Linear): Key projection
        value (nn.Linear): Value projection
        query (nn.Linear): Query projection
        proj (nn.Linear): Output projection
        attn_dropout (nn.Dropout): Attention dropout
        proj_dropout (nn.Dropout): Output dropout
        mask (torch.Tensor): Causal attention mask
        k_cache (torch.Tensor): Key cache
        v_cache (torch.Tensor): Value cache
        current_length (int): Current cache position
    """
    
    def __init__(self, hparams):
        """Initialize attention module.
        
        Args:
            hparams (HyperParameters): Model configuration
        """
        super().__init__()
        self.num_heads = hparams.num_heads
        self.head_dim = hparams.embed_dim // self.num_heads
        
        # Projection layers
        self.key = nn.Linear(hparams.embed_dim, hparams.embed_dim)
        self.value = nn.Linear(hparams.embed_dim, hparams.embed_dim)
        self.query = nn.Linear(hparams.embed_dim, hparams.embed_dim)
        self.proj = nn.Linear(hparams.embed_dim, hparams.embed_dim)
        
        # Dropout layers
        self.attn_dropout = nn.Dropout(hparams.attn_dropout)
        self.proj_dropout = nn.Dropout(hparams.ff_dropout)
        
        # Causal mask and KV cache
        self.register_buffer("mask", torch.triu(
            torch.ones(hparams.max_len, hparams.max_len).bool(), diagonal=1
        ).unsqueeze(0).unsqueeze(0))
        
        # Cache buffers
        self.register_buffer("k_cache", torch.zeros(
            hparams.batch_size, self.num_heads, hparams.max_len, self.head_dim
        ))
        self.register_buffer("v_cache", torch.zeros(
            hparams.batch_size, self.num_heads, hparams.max_len, self.head_dim
        ))
        self.current_length = 0

    def forward(self, x, use_cache=False):
        """Compute attention with optional caching.
        
        Args:
            x (torch.Tensor): Input tensor [batch_size, seq_len, embed_dim]
            use_cache (bool): Use KV caching
            
        Returns:
            torch.Tensor: Attention output [batch_size, seq_len, embed_dim]
        """
        batch_size, seq_len, _ = x.size()
        
        # Project inputs
        q = self._reshape(self.query(x))
        new_k = self._reshape(self.key(x))
        new_v = self._reshape(self.value(x))
        
        # Update cache
        if use_cache:
            self._update_cache(new_k, new_v, seq_len)
            self.current_length += seq_len
            k = self.k_cache[:, :, :self.current_length, :]
            v = self.v_cache[:, :, :self.current_length, :]
        else:
            self._reset_cache(new_k, new_v, seq_len)
            k, v = new_k, new_v
        
        # Compute attention with masking
        
        mask = self.mask[:, :, self.current_length-seq_len:self.current_length, :self.current_length]
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        
        # Process output
        y = y.transpose(1, 2).reshape(batch_size, seq_len, -1)
        return self.proj_dropout(self.proj(y))
    
    def _reshape(self, x):
        """Reshape tensor for multi-head computation."""
        return x.view(x.size(0), x.size(1), self.num_heads, self.head_dim).transpose(1, 2)
    
    def _update_cache(self, new_k, new_v, seq_len):
        """Update KV cache for autoregressive generation."""
        self.k_cache[:, :, self.current_length:self.current_length+seq_len] = new_k
        self.v_cache[:, :, self.current_length:self.current_length+seq_len] = new_v
        
    def _reset_cache(self, new_k, new_v, seq_len):
        """Reset cache for new sequence."""
        self.k_cache[:, :, :seq_len] = new_k
        self.v_cache[:, :, :seq_len] = new_v
        self.current_length = seq_len