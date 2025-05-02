# coding=utf-8
# Required Libraries:
# pip install torch transformers accelerate bitsandbytes einops causal-conv1d mamba-ssm packaging datasets sentencepiece

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from typing import List, Dict, Tuple, Optional, Union

from transformers import PreTrainedModel, AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.utils import logging
from transformers.modeling_outputs import ModelOutput, CausalLMOutputWithPast
from transformers.models.mamba.configuration_mamba import MambaConfig as BaseMambaConfig
from transformers.cache_utils import Cache, DynamicCache
from transformers.models.mamba.modeling_mamba import (
    MambaBlock, # Direct import
    MambaMixer,
    MambaRMSNorm,
    MambaPreTrainedModel,
    MambaCache
)
from transformers.generation import GenerationMixin
from transformers.utils import logging

import mathtools # Used via \DeclareMathOperator in preamble
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, random_split
import os # For checking directory existence

# Setup Logging
logger = logging.get_logger(__name__)
logging.set_verbosity_info()

# Define Math Operators for Loss Function
@torch.no_grad() # Ensure no gradients for argmax
def _argmax_ignore_grad(logits):
    return torch.argmax(logits, dim=-1)

def _calculate_mse_loss(input_tensor, target_tensor):
    target_tensor = target_tensor.to(input_tensor.dtype)
    if target_tensor.shape != input_tensor.shape:
        if target_tensor.dim() == input_tensor.dim() - 1 and input_tensor.shape[-1] == 1:
            target_tensor = target_tensor.unsqueeze(-1)
        else:
             # Try squeezing target if input is scalar-like per position
             if input_tensor.shape == target_tensor.shape + (1,):
                 input_tensor = input_tensor.squeeze(-1)
             elif target_tensor.shape == input_tensor.shape + (1,):
                 target_tensor = target_tensor.squeeze(-1)
             else:
                 raise ValueError(f"MSE Target shape {target_tensor.shape} incompatible with input shape {input_tensor.shape}")
    return F.mse_loss(input_tensor, target_tensor, reduction='mean')

def _calculate_kl_loss(log_probs_student, probs_teacher, temperature):
     # KLDivLoss expects log-probabilities for input, probabilities for target.
     # reduction='batchmean' averages over batch and sequence dimensions THEN the class probabilities.
     kd_loss = F.kl_div(log_probs_student, probs_teacher, reduction='batchmean', log_target=False)
     # Scale by T^2
     kd_loss = kd_loss * (temperature ** 2)
     return kd_loss

# === 1. Auxiliary Modules ===

class ExitHead(nn.Module):
    """Simple linear classifier for intermediate exits."""
    def __init__(self, d_model: int, vocab_size: int, bias: bool = False):
        super().__init__()
        self.classifier = nn.Linear(d_model, vocab_size, bias=bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """ Returns logits. """
        return self.classifier(hidden_states)

class SharedConfidenceNetwork(nn.Module):
    """Shared network to predict confidence score based on hidden state."""
    def __init__(self, d_model: int, bias: bool = True):
        super().__init__()
        self.fc = nn.Linear(d_model, 1, bias=bias) # Outputs a single logit

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """ Returns a single logit per token position. Apply sigmoid later. """
        return self.fc(hidden_states)

# === 2. Student Model Config and Class ===

class EarlyExitMambaConfig(BaseMambaConfig): # Inherit from base Mamba Config
    model_type = "early_exit_mamba" # Define a unique model_type

    def __init__(self, add_intermediate_norms: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.add_intermediate_norms = add_intermediate_norms

class EarlyExitMambaKD(MambaPreTrainedModel, GenerationMixin):
    config_class = EarlyExitMambaConfig
    _tied_weights_keys = ["lm_head.classifier.weight"]

    def __init__(self, config: EarlyExitMambaConfig):
        super().__init__(config)
        self.config = config
        self.hidden_size = config.hidden_size
        self.n_layers = config.num_hidden_layers
        self.vocab_size = config.vocab_size
        self.use_bias = config.use_bias
        self.add_intermediate_norms = config.add_intermediate_norms

        logger.info(f"Initializing EarlyExitMambaKD model with {self.n_layers} layers.")
        logger.info("Attaching early exit heads after EVERY intermediate layer (1 to N-1).")

        # --- Core Mamba Components ---
        self.embeddings = nn.Embedding(config.vocab_size, self.hidden_size)
        self.layers = nn.ModuleList([
            MambaBlock(config, layer_idx=idx) for idx in range(config.num_hidden_layers)
        ])
        self.intermediate_norms = nn.ModuleDict() if self.add_intermediate_norms else None
        if self.add_intermediate_norms:
            for i in range(1, self.n_layers): # Norms before intermediate heads
                self.intermediate_norms[str(i)] = MambaRMSNorm(self.hidden_size, eps=config.layer_norm_epsilon)
        self.norm_f = MambaRMSNorm(self.hidden_size, eps=config.layer_norm_epsilon) # Final norm
        # --- End Core Mamba ---

        # --- Early Exit Components ---
        self.exit_heads = nn.ModuleDict()
        for i in range(1, self.n_layers): # Exits AFTER layers 1 to N-1
            self.exit_heads[str(i)] = ExitHead(self.hidden_size, self.vocab_size, bias=self.use_bias)
        logger.info(f"Created {len(self.exit_heads)} intermediate exit heads.")
        if self.add_intermediate_norms:
             logger.info("Added RMSNorm before each intermediate exit head.")

        # Final layer classifier (used for layer N output)
        self.lm_head = ExitHead(self.hidden_size, self.vocab_size, bias=self.use_bias)
        logger.info("Created final LM head.")

        # Shared Confidence Network (used by all N heads)
        self.confidence_network = SharedConfidenceNetwork(self.hidden_size, bias=True)
        # --- End Early Exit Components ---

        # Initialize weights and apply final processing
        self.post_init()

    # --- HF compatibility methods ---
    def get_input_embeddings(self): return self.embeddings
    def set_input_embeddings(self, new_embeddings): self.embeddings = new_embeddings
    def get_output_embeddings(self): return self.lm_head.classifier
    def set_output_embeddings(self, new_embeddings): self.lm_head.classifier = new_embeddings

    def _update_model_kwargs_for_generation(self, outputs, model_kwargs, num_new_tokens=1, **kwargs):
        # Standard Mamba cache update logic
        model_kwargs["cache_params"] = outputs.get("cache_params", None)
        if model_kwargs.get("use_cache", True) and "cache_position" in model_kwargs and model_kwargs["cache_position"] is not None:
            model_kwargs["cache_position"] = model_kwargs["cache_position"][-1:] + num_new_tokens
        return model_kwargs

    def prepare_inputs_for_generation(self, input_ids, inputs_embeds=None, use_cache=None, cache_params=None, cache_position=None, **kwargs):
         # Standard Mamba input prep logic
         if use_cache and cache_params is not None and cache_position is not None:
              if cache_position[0] > 0:
                  input_ids = input_ids[:, -1].unsqueeze(-1)
         if inputs_embeds is not None and cache_params is None:
             model_inputs = {"inputs_embeds": inputs_embeds}
         else:
             model_inputs = {"input_ids": input_ids.contiguous()}
         model_inputs.update({"cache_params": cache_params, "use_cache": use_cache, "cache_position": cache_position})
         return model_inputs
    # --- End HF compatibility methods ---

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_params: Optional[MambaCache] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None, # Accept attention mask although not used by MambaBlock
        return_all_outputs: bool = False
    ) -> Union[Tuple, Dict, CausalLMOutputWithPast]:

        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)

        if use_cache and cache_params is None:
             # Initialize cache if use_cache is True and no cache is provided
             cache_params = MambaCache(self.config, inputs_embeds.size(0), device=inputs_embeds.device, dtype=inputs_embeds.dtype)

        if cache_position is None:
             # Initialize cache_position if not provided
             if use_cache:
                 cache_position = torch.arange(cache_params.seqlen_offset, cache_params.seqlen_offset + inputs_embeds.shape[1], device=inputs_embeds.device)
             else:
                 cache_position = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)

        hidden_states = inputs_embeds
        all_layer_hidden_states_list = [hidden_states] if output_hidden_states else None
        all_exit_logits = {}
        all_confidence_logits = {}

        for i in range(self.n_layers):
            layer_idx_str = str(i + 1)
            # Pass hidden_states through the Mamba layer
            # Note: HF MambaBlock internally handles pre-LayerNorm (self.norm)
            layer_outputs = self.layers[i](
                hidden_states, cache_params=cache_params, cache_position=cache_position
            )
            hidden_states = layer_outputs[0] if isinstance(layer_outputs, tuple) else layer_outputs

            if output_hidden_states:
                all_layer_hidden_states_list.append(hidden_states)

            # --- Calculate intermediate or final outputs ---
            # State prepared for heads (potentially normalized)
            state_for_heads = hidden_states
            if i + 1 < self.n_layers: # Intermediate layers
                if self.add_intermediate_norms:
                     state_for_heads = self.intermediate_norms[layer_idx_str](hidden_states)
                exit_logits = self.exit_heads[layer_idx_str](state_for_heads)
            else: # Final layer N
                 state_for_heads = self.norm_f(hidden_states) # Apply final norm
                 exit_logits = self.lm_head(state_for_heads)

            conf_logits = self.confidence_network(state_for_heads) # Confidence uses same state

            all_exit_logits[layer_idx_str] = exit_logits
            all_confidence_logits[layer_idx_str] = conf_logits

            # Update cache_position only if use_cache is True
            if use_cache:
                cache_position = cache_position + 1 # Assumes generation step (len=1)

        # --- Prepare return values ---
        if output_hidden_states:
             all_hidden_states = tuple(all_layer_hidden_states_list)
        else:
             all_hidden_states = None

        final_lm_logits = all_exit_logits[str(self.n_layers)] # Get final logits

        if return_all_outputs:
            outputs = {
                "exit_logits": all_exit_logits,
                "confidence_logits": all_confidence_logits,
                "last_hidden_state": state_for_heads, # State after final norm
                "cache_params": cache_params if use_cache else None,
                "hidden_states": all_hidden_states,
            }
            return outputs if not return_dict else ModelOutput(**outputs)
        else:
            return CausalLMOutputWithPast(
                loss=None,
                logits=final_lm_logits,
                past_key_values=cache_params,
                hidden_states=all_hidden_states,
                attentions=None,
            )


# === 3. Initialization Function (Handles shape mismatches heuristically) ===
@torch.no_grad()
@torch.no_grad()
def initialize_student_from_teacher(student_model: EarlyExitMambaKD, teacher_model_path: str):
    """Initializes student Mamba parameters from teacher Transformer using attention-to-Mamba mapping theory."""
    logger.info(f"Initializing student from {teacher_model_path} using Attention-to-Mamba mapping...")

    try:
        teacher = AutoModelForCausalLM.from_pretrained(teacher_model_path, trust_remote_code=True).eval()
        teacher_hidden_size = teacher.config.hidden_size
        student_hidden_size = student_model.config.hidden_size

        # Verify dimension compatibility
        if teacher_hidden_size != student_hidden_size:
            raise ValueError(f"Teacher hidden size {teacher_hidden_size} != student {student_hidden_size}")

        # Layer mapping
        num_teacher_layers = teacher.config.num_hidden_layers
        num_student_layers = student_model.config.num_hidden_layers
        layer_ratio = num_teacher_layers // num_student_layers

        for student_idx, student_layer in enumerate(student_model.layers):
            # Get corresponding teacher layer
            teacher_idx = student_idx * layer_ratio
            teacher_layer = teacher.model.layers[teacher_idx]
            mamba_mixer = student_layer.mixer

            # Get SSM parameters
            d_model = student_model.config.hidden_size
            expand = mamba_mixer.d_inner // d_model  # Expansion factor
            d_state = mamba_mixer.d_state

            # 1. Initialize in_proj (V mapping)
            with torch.no_grad():
                # Teacher projections
                W_v = teacher_layer.self_attn.v_proj.weight  # (d_model, d_model)

                # Student in_proj (first half for x)
                in_proj_weight = mamba_mixer.in_proj.weight
                d_inner = mamba_mixer.d_inner
                x_proj_weight = in_proj_weight[:d_inner]  # (d_inner, d_model)

                # Tile teacher's V projections to match expanded dimension
                x_proj_weight.copy_(W_v.repeat(expand, 1))

                # 2. Initialize B from K and C from Q (x_proj mapping)
                W_k = teacher_layer.self_attn.k_proj.weight[:d_state]  # (d_state, d_model)
                W_q = teacher_layer.self_attn.q_proj.weight[:d_state]  # (d_state, d_model)

                # Get x_proj weights and split into delta/B/C
                x_proj = mamba_mixer.x_proj.weight
                d_conv = mamba_mixer.d_conv
                B_slice = slice(d_conv, d_conv+d_state)
                C_slice = slice(d_conv+d_state, d_conv+2*d_state)

                # Expand K/Q projections to match student dimension
                x_proj[B_slice] = W_k.repeat(1, expand)  # (d_state, d_inner)
                x_proj[C_slice] = W_q.repeat(1, expand)

            logger.info(f"Mapped teacher layer {teacher_idx} to student layer {student_idx}")

        logger.info("Successfully initialized student with teacher projections!")

    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        logger.warning("Proceeding with default initialization")

# === 5. Loss Function (Combined) ===
# (Same as previous correct version)
# ... definition of calculate_combined_loss ...
@torch.no_grad()
def _argmax_ignore_grad(logits): return torch.argmax(logits, dim=-1)
def _calculate_mse_loss(pred, target): return F.mse_loss(pred, target.to(pred.dtype), reduction='mean')
def _calculate_kl_loss(log_p_student, p_teacher, temp): return F.kl_div(log_p_student, p_teacher, reduction='batchmean', log_target=False) * (temp**2)

def calculate_combined_loss(student_outputs, teacher_logits, labels, loss_weights, temperature, task_type='classification'):
    total_loss = 0.0
    exit_layer_indices = sorted([int(k) for k in student_outputs["exit_logits"].keys()])
    N = exit_layer_indices[-1]
    shift_labels = labels[:, 1:].contiguous()
    shift_labels_flat = shift_labels.view(-1)
    shift_teacher_logits = None
    if teacher_logits is not None and loss_weights.get('lambda_kd', 0) > 0:
        shift_teacher_logits = teacher_logits[:, :-1, :].contiguous().to(torch.float32)
    vocab_size = -1

    for i, layer_idx in enumerate(exit_layer_indices):
        layer_idx_str = str(layer_idx)
        s_logits_full = student_outputs["exit_logits"][layer_idx_str]
        conf_logits_full = student_outputs["confidence_logits"][layer_idx_str]
        s_logits = s_logits_full[:, :-1, :].contiguous()
        conf_logits = conf_logits_full[:, :-1, :].contiguous()
        if vocab_size == -1: vocab_size = s_logits.size(-1)

        ce_loss = F.cross_entropy(s_logits.view(-1, vocab_size), shift_labels_flat, reduction='mean')
        lambda_i = loss_weights.get(f'lambda_ce_{layer_idx_str}', loss_weights.get('lambda_ce', 1.0))
        layer_loss_contribution = lambda_i * ce_loss

        pred_tokens = _argmax_ignore_grad(s_logits)
        is_correct = (pred_tokens == shift_labels).float()
        u_tilde_i = is_correct
        u_i = torch.sigmoid(conf_logits)
        conf_loss = _calculate_mse_loss(u_i.squeeze(-1), u_tilde_i)
        layer_loss_contribution += loss_weights['lambda_conf'] * conf_loss

        if layer_idx < N and shift_teacher_logits is not None:
            log_p_student = F.log_softmax(s_logits.to(torch.float32) / temperature, dim=-1)
            p_teacher = F.softmax(shift_teacher_logits / temperature, dim=-1)
            kd_loss = _calculate_kl_loss(log_p_student, p_teacher, temperature)
            layer_loss_contribution += loss_weights['lambda_kd'] * kd_loss
        total_loss += layer_loss_contribution
    return total_loss

# === 6. Training Loop Function ===
# (Same as previous correct version)
# ... definition of train_epoch ...
def train_epoch(model, teacher_model, dataloader, optimizer, scheduler, loss_weights, temperature, device, current_epoch, task_type='classification'):
    model.train()
    if teacher_model: teacher_model.eval()
    total_epoch_loss = 0.0
    num_batches = len(dataloader)
    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch['input_ids'].to(device)
        labels = batch.get('labels', input_ids).to(device)
        optimizer.zero_grad()
        student_outputs = model(input_ids, return_all_outputs=True)
        teacher_logits = None
        if teacher_model and loss_weights.get('lambda_kd', 0) > 0:
            with torch.no_grad():
                teacher_outputs = teacher_model(input_ids)
                teacher_logits = teacher_outputs.logits
        # Alternating strategy
        is_alternating = loss_weights.get('alternating', False)
        is_odd_epoch = (current_epoch % 2 == 1)
        if is_alternating and is_odd_epoch:
            final_layer_idx_str = str(model.n_layers)
            final_logits_full = student_outputs["exit_logits"][final_layer_idx_str]
            final_logits = final_logits_full[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(final_logits.view(-1, final_logits.size(-1)), shift_labels.view(-1))
        else:
            loss = calculate_combined_loss(student_outputs, teacher_logits, labels, loss_weights, temperature, task_type)
        loss.backward()
        optimizer.step()
        if scheduler: scheduler.step()
        total_epoch_loss += loss.item()
        if batch_idx % 10 == 0: # Log progress
            print(f"  Epoch {current_epoch} Batch {batch_idx}/{num_batches}, Batch Loss: {loss.item():.4f}")
    avg_loss = total_epoch_loss / num_batches
    print(f"Epoch {current_epoch} finished. Average Loss: {avg_loss:.4f}")
    return avg_loss

# === 7. Inference Function ===
# (Same as previous correct version)
# ... definition of inference_with_early_exit ...
@torch.no_grad()
def inference_with_early_exit(model, input_ids, threshold, device):
    model.eval()
    batch_size, seq_len = input_ids.shape
    # Initialize cache and position for generation simulation
    # Note: Actual generation uses prepare_inputs_for_generation, this is simplified
    hidden_states = model.embeddings(input_ids.to(device))
    cache_params = MambaCache(model.config, batch_size, device=device, dtype=hidden_states.dtype)
    cache_position = torch.arange(seq_len, device=device)

    for i in range(model.n_layers):
        layer_outputs = model.layers[i](hidden_states, cache_params=cache_params, cache_position=cache_position[:hidden_states.shape[1]])
        if isinstance(layer_outputs, tuple): hidden_states = layer_outputs[0]
        else: hidden_states = layer_outputs
        cache_position = cache_position + hidden_states.shape[1] # Simplistic update

        current_layer_idx = i + 1
        current_layer_idx_str = str(current_layer_idx)
        state_for_heads = hidden_states
        if current_layer_idx < model.n_layers:
            if model.add_intermediate_norms:
                state_for_heads = model.intermediate_norms[current_layer_idx_str](hidden_states)
        else:
            state_for_heads = model.norm_f(hidden_states)

        conf_logits = model.confidence_network(state_for_heads[:, -1:, :])
        confidence_score = torch.sigmoid(conf_logits).squeeze().item()

        print(f"  Layer {current_layer_idx} Confidence: {confidence_score:.4f}")

        if confidence_score > threshold:
            if current_layer_idx < model.n_layers:
                 exit_logits_full = model.exit_heads[current_layer_idx_str](state_for_heads)
            else:
                 exit_logits_full = model.lm_head(state_for_heads)
            exit_logits_last_token = exit_logits_full[:, -1, :]
            print(f"  Early exiting at layer {current_layer_idx}")
            return exit_logits_last_token, current_layer_idx

    # Fallback if no exit triggered (should ideally exit at N)
    final_layer_idx_str = str(model.n_layers)
    # Recompute final logits if needed (although they should be in all_exit_logits if forward was run fully)
    state_final_norm = model.norm_f(hidden_states)
    final_logits_full = model.lm_head(state_final_norm)
    final_logits_last_token = final_logits_full[:, -1, :]
    print(f"  Exiting at final layer {model.n_layers} (fallback)")
    return final_logits_last_token, model.n_layers


# === 8. Main Execution Block ===
if __name__ == "__main__":
    # --- Configuration ---
    # Match TinyLlama 1.1B config as much as possible for Mamba
    teacher_model_name = "EleutherAI/pythia-70m"
    logger.info("Loading teacher config to align student config...")
    teacher_config = AutoConfig.from_pretrained(teacher_model_name, trust_remote_code=True)

    student_config_dict = {
        'hidden_size': teacher_config.hidden_size,         # Match hidden size
        'n_layer': teacher_config.num_hidden_layers,   # Match layer count
        'vocab_size': teacher_config.vocab_size,       # Match vocab size
        'ssm_cfg': {'d_state': 16, 'd_conv': 4, 'expand': 2}, # Mamba specific - KEEP THESE SMALL FOR COLAB
        'rms_norm': True, 'residual_in_fp32': True, 'fused_add_norm': True,
        'pad_vocab_size_multiple': 8, 'use_bias': False,
        'layer_norm_epsilon': getattr(teacher_config, "rms_norm_eps", 
                                  getattr(teacher_config, "layer_norm_eps", 1e-5)),
        'hidden_act': teacher_config.hidden_act,
        'initializer_range': teacher_config.initializer_range,
        'intermediate_size': teacher_config.intermediate_size, # Match intermediate size
        'time_step_rank': 'auto',
        'time_step_scale': 1.0, 'time_step_min': 0.001, 'time_step_max': 0.1,
        'time_step_init_scheme': 'random', 'time_step_floor': 1e-4,
        'use_conv_bias': True, # Mamba default
        'rescale_prenorm_residual': False,
        'num_hidden_layers': teacher_config.num_hidden_layers, # Explicitly set for MambaConfig base
        # --- Early Exit Config ---
        'add_intermediate_norms': True,
    }
    student_config = EarlyExitMambaConfig(**student_config_dict)

    loss_weights = { 'lambda_ce': 1.0, 'lambda_conf': 0.1, 'lambda_kd': 0.5, 'alternating': False } # Adjusted example weights
    temperature = 2.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 1 # Keep low for testing
    learning_rate = 1e-5 # Lower LR often needed for fine-tuning/distillation
    inference_threshold = 0.85 # Example

    # --- Load Teacher ---
    print(f"Loading teacher: {teacher_model_name}")
    try:
        teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model_name, trust_remote_code=True).to(device)
        teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_name, trust_remote_code=True)
        for param in teacher_model.parameters(): param.requires_grad = False
        # Update student config vocab size *again* just in case tokenizer changed it
        student_config.vocab_size = teacher_model.config.vocab_size
        student_config.pad_token_id = teacher_tokenizer.pad_token_id if teacher_tokenizer.pad_token_id is not None else teacher_tokenizer.eos_token_id
        if teacher_tokenizer.pad_token is None:
            logger.warning("Tokenizer missing pad token, using EOS token as pad token.")
            teacher_tokenizer.pad_token = teacher_tokenizer.eos_token

        print("Teacher model loaded and frozen.")
    except Exception as e:
        print(f"Error loading teacher model {teacher_model_name}: {e}")
        exit()

    # --- Init Student ---
    print("Initializing student model...")
    student_model = EarlyExitMambaKD(student_config).to(device)
    student_params = sum(p.numel() for p in student_model.parameters() if p.requires_grad) / 1e6
    print(f"Student model initialized with {student_params:.2f}M trainable parameters.")
    # Note: Total params will be slightly higher due to non-Mamba components (Embeddings, Norms, Heads)

    # --- Init Weights ---
    initialize_student_from_teacher(student_model, teacher_model_name)

    # --- Prepare Dummy Data ---
    print("Preparing dummy data...")
    dummy_seq_len = 64 # Shorter sequence for faster testing
    dummy_dataset = [{'input_ids': torch.randint(0, student_config.vocab_size, (dummy_seq_len,)),
                      'labels': torch.randint(0, student_config.vocab_size, (dummy_seq_len,))} for _ in range(20)] # More batches
    class DummyDataset(Dataset):
        def __init__(self, data): self.data = data
        def __len__(self): return len(self.data)
        def __getitem__(self, idx): return self.data[idx]
    # Collate function to handle padding if necessary (though Mamba might not need strict padding)
    def collate_fn(batch):
        input_ids = [item['input_ids'] for item in batch]
        labels = [item['labels'] for item in batch]
        # Pad to max length in batch
        input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=teacher_tokenizer.pad_token_id or 0)
        labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100) # Use -100 for ignored labels
        return {'input_ids': input_ids_padded, 'labels': labels_padded}

    dummy_dataloader = DataLoader(DummyDataset(dummy_dataset), batch_size=4, collate_fn=collate_fn)
    print("Dummy data ready.")

    # --- Optimizer ---
    optimizer = AdamW(student_model.parameters(), lr=learning_rate)
    scheduler = None

    # --- Training ---
    print(f"\n--- Training ---")
    for epoch in range(num_epochs):
        train_epoch(student_model, teacher_model, dummy_dataloader, optimizer, scheduler,
                    loss_weights, temperature, device, epoch + 1, task_type='classification')
    print("--- Training finished ---")

    # --- Inference ---
    print("\n--- Inference Example ---")
    test_input_text = "Alan Turing was a"
    # Ensure input is tokenized correctly
    test_input_ids = teacher_tokenizer(test_input_text, return_tensors="pt")['input_ids'].to(device)

    final_logits, exited_at_layer = inference_with_early_exit(student_model, test_input_ids, inference_threshold, device)
    predicted_next_token_id = torch.argmax(final_logits, dim=-1).item()
    predicted_token = teacher_tokenizer.decode(predicted_next_token_id)
    print(f"Input: '{test_input_text}'")
    print(f"Exited at layer: {exited_at_layer}")
    print(f"Predicted next token: '{predicted_token}' (ID: {predicted_next_token_id})")
    print("--- Inference finished ---")