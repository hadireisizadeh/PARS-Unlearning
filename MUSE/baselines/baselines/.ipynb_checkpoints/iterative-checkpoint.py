from .utils import load_model_and_tokenizer, load_model
from .dataset import ForgetRetainDataset

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda import device_count
import transformers
from transformers import Trainer, AutoModelForCausalLM
import copy
from accelerate.hooks import remove_hook_from_module
from typing import List, Optional

class ProbeDecoder(nn.Module):
    
    def __init__(self, source_model, probe_device: int = 7):
        super().__init__()
        self.decoder_device = torch.device(
            f"cuda:{probe_device}" if torch.cuda.is_available() else "cpu"
        )
        norm    = copy.deepcopy(source_model.model.norm)
        lm_head = copy.deepcopy(source_model.lm_head)
        remove_hook_from_module(norm,    recurse=True)
        remove_hook_from_module(lm_head, recurse=True)
        self.norm    = norm.float().to(self.decoder_device)
        self.lm_head = lm_head.float().to(self.decoder_device)

    def forward(self, hidden):
        hidden = hidden.to(self.decoder_device).float()
        return self.lm_head(self.norm(hidden))


# def get_hidden_state_with_grad(model, input_ids, layer_idx):

#     embed_device = next(model.model.embed_tokens.parameters()).device
#     hidden = model.model.embed_tokens(input_ids.to(embed_device))

#     seq_len = hidden.shape[1]
#     position_ids = torch.arange(seq_len, device=hidden.device).unsqueeze(0)

#     # Pre-compute rotary embeddings — required by newer transformers LLaMA
#     position_embeddings = None
#     if hasattr(model.model, 'rotary_emb'):
#         position_embeddings = model.model.rotary_emb(hidden, position_ids)

#     for i, layer in enumerate(model.model.layers):
#         if i > layer_idx:
#             break
#         layer_device = next(layer.parameters()).device
#         hidden = hidden.to(layer_device)

#         layer_kwargs = {
#             'use_cache': False,
#             'position_ids': position_ids.to(layer_device),
#         }
#         if position_embeddings is not None:
#             cos, sin = position_embeddings
#             layer_kwargs['position_embeddings'] = (
#                 cos.to(layer_device), sin.to(layer_device)
#             )

#         hidden = layer(hidden, **layer_kwargs)[0]

#     return hidden.float()

def get_hidden_state_with_grad(model, input_ids, layer_idx):
    embed_device = next(model.model.embed_tokens.parameters()).device
    hidden = model.model.embed_tokens(input_ids.to(embed_device))
    for i, layer in enumerate(model.model.layers):
        if i > layer_idx:
            break
        layer_device = next(layer.parameters()).device
        hidden = hidden.to(layer_device)
        position_ids = torch.arange(
            hidden.shape[1], device=layer_device
        ).unsqueeze(0)
        position_embeddings = model.model.rotary_emb(hidden, position_ids)
        layer_output = layer(
            hidden,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            use_cache=False,
        )
        hidden = layer_output[0] if isinstance(layer_output, tuple) else layer_output
    #
    return hidden.float()


def get_hidden_state_no_grad(model, input_ids, layer_idx):
    with torch.no_grad():
        h = get_hidden_state_with_grad(model, input_ids, layer_idx)
        return h.detach()


def compute_probe_loss(
    probe:   ProbeDecoder,
    hidden:  torch.Tensor,   
    labels:  torch.Tensor,   
) -> torch.Tensor:
    
    logits = probe(hidden)                              
    labels = labels.to(probe.decoder_device)

    shift_logits = logits[..., :-1, :].contiguous()    
    shift_labels = labels[..., 1:].contiguous()         

    loss = nn.CrossEntropyLoss(ignore_index=-100)(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )
    return loss


def unlearn_probe(
    model_dir:              str,
    data_file:              str,
    out_dir:                str,
    retain_data_file:       str | None  = None,
    loss_type:              str         = 'ga',
    per_device_batch_size:  int         = 1,
    epochs:                 int         = 5,
    learning_rate:          float       = 1e-5,
    max_len:                int         = 1024,
    tokenizer_dir:          str | None  = None,
    resume_from_checkpoint: bool        = False,
    retain_alpha:           float       = 1.0,
    probe_layers:           List[int]   = None,
    probe_lr:               float       = 1e-4,
    probe_inner_steps:      int         = 3,
    probe_beta:             float       = 0.5,
    probe_device:           int         = 7,  # Probe runs on separate GPU
    keep_checkpoints:       int         = 1,
    probe_r_f_both:         bool        = False,  # Use both forget (+) and retain (-) in probe loss
):
    print("loss type: ",loss_type)
    print("batch size: ",per_device_batch_size)
    if 'gd' in loss_type:
        assert retain_data_file is not None, \
            "Retain data must be specified for grad_diff."

    model, tokenizer = load_model_and_tokenizer(
        model_dir,
        tokenizer_dir=tokenizer_dir,
    )

    ref_model = (
        load_model(model_dir)
        if 'npo' in loss_type or 'kl' in loss_type
        else None
    )

    dataset = ForgetRetainDataset(
        data_file,
        tokenizer=tokenizer,
        retain_file_path=retain_data_file,
        max_len=max_len,
    )

    if device_count() == 0:
        raise ValueError("Device not detected!")

    training_args = transformers.TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=per_device_batch_size,
        learning_rate=learning_rate,
        save_strategy='epoch',
        num_train_epochs=epochs,
        optim='adamw_torch',
        lr_scheduler_type='constant',
        bf16=True,
        report_to='none',
        gradient_accumulation_steps=8,  # Effective batch size = 8
        gradient_checkpointing=True,    # Enable gradient checkpointing
    )

    trainer = ProbeUnlearner(
        model=model,
        ref_model=ref_model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=training_args,
        data_collator=dataset.get_collate_fn(),
        loss_type=loss_type,
        retain_alpha=retain_alpha,
        probe_layers=probe_layers or [],
        probe_lr=probe_lr,
        probe_inner_steps=probe_inner_steps,
        probe_beta=probe_beta,
        probe_device=probe_device,
        probe_r_f_both=probe_r_f_both,
    )

    model.config.use_cache = False
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(out_dir)


class ProbeUnlearner(Trainer):

    def __init__(
        self,
        *args,
        loss_type:         str             = 'ga',
        ref_model:         AutoModelForCausalLM | None = None,
        beta:              float           = 0.1,
        retain_alpha:      float           = 1.0,
        probe_layers:      List[int]       = None,
        probe_lr:          float           = 1e-4,
        probe_inner_steps: int             = 3,
        probe_beta:        float           = 0.5,
        probe_device:      int             = 7,  # Probe runs on separate GPU
        probe_r_f_both:    bool            = False,  # Use both forget (+) and retain (-) in probe loss
        **kwargs,
    ):
        self.loss_type         = loss_type
        self.ref_model         = ref_model
        self.beta              = beta              # NPO temperature
        self.retain_alpha      = retain_alpha      # weight of retain loss term
        self.probe_layers      = probe_layers or []
        self.probe_lr          = probe_lr
        self.probe_inner_steps = probe_inner_steps
        self.probe_beta        = probe_beta        # weight of probe term
        self.probe_device      = probe_device      # Probe GPU device
        self.probe_r_f_both    = probe_r_f_both    # Use both forget and retain in probe loss

        if ref_model is not None:
            assert 'po' in self.loss_type or 'kl' in self.loss_type
            ref_model = ref_model.eval()

        super().__init__(*args, **kwargs)

        if self.probe_layers:
            self.probes = {
                ell: ProbeDecoder(self.model, probe_device=self.probe_device)
                for ell in self.probe_layers
            }
            
            self.probe_optimizers = {
                ell: torch.optim.AdamW(
                    self.probes[ell].parameters(), lr=self.probe_lr
                )
                for ell in self.probe_layers
            }
        else:
            self.probes           = {}
            self.probe_optimizers = {}

    #def compute_loss(self, model, x, return_outputs=False):
    def compute_loss(self, model, x, return_outputs=False, **kwargs):

        x_f, x_r = x

        forget_ids    = x_f['input_ids']
        forget_labels = x_f['labels'] if 'labels' in x_f \
                        else x_f['input_ids'].clone()
        forget_mask   = x_f['attention_mask'] if 'attention_mask' in x_f \
                        else torch.ones_like(x_f['input_ids'], dtype=torch.bool)

        outputs_f = model(
            forget_ids,
            labels=forget_labels,
            attention_mask=forget_mask,
        )
        loss_f = outputs_f.loss

        # Extract retain data if needed for loss_type or probe_r_f_both
        if 'gdr' in self.loss_type or 'klr' in self.loss_type or self.probe_r_f_both:
            retain_ids    = x_r['input_ids']
            retain_labels = x_r['labels'] if 'labels' in x_r \
                            else x_r['input_ids'].clone()
            retain_mask   = x_r['attention_mask'] if 'attention_mask' in x_r \
                            else torch.ones_like(x_r['input_ids'], dtype=torch.bool)

        if 'gdr' in self.loss_type or 'klr' in self.loss_type:
            outputs_r = model(
                retain_ids,
                labels=retain_labels,
                attention_mask=retain_mask,
            )
            loss_r = outputs_r.loss

        if 'klf' in self.loss_type or 'npo' in self.loss_type:
            with torch.no_grad():
                outputs_f_ref = self.ref_model(
                    forget_ids,
                    labels=forget_labels,
                    attention_mask=forget_mask,
                )

        if 'klr' in self.loss_type:
            with torch.no_grad():
                outputs_r_ref = self.ref_model(
                    retain_ids,
                    labels=retain_labels,
                    attention_mask=retain_mask,
                )

        if self.probe_layers:
            for p in model.parameters():
                p.requires_grad_(False)

            for _ in range(self.probe_inner_steps):
                for ell in self.probe_layers:
                    probe = self.probes[ell]
                    probe.train()
                    self.probe_optimizers[ell].zero_grad()

                    hidden = get_hidden_state_no_grad(
                        model, forget_ids, ell
                    )
                    loss_probe_inner = compute_probe_loss(
                        probe, hidden, forget_labels
                    )

                    if self.probe_r_f_both:
                        hidden_r = get_hidden_state_no_grad(
                            model, retain_ids, ell
                        )
                        loss_probe_inner_r = compute_probe_loss(
                            probe, hidden_r, retain_labels
                        )
                        
                        loss_probe_inner = loss_probe_inner - loss_probe_inner_r

                    loss_probe_inner.backward()
                    nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
                    self.probe_optimizers[ell].step()

            for p in model.parameters():
                p.requires_grad_(True)

            for ell in self.probe_layers:
                for p in self.probes[ell].parameters():
                    p.requires_grad_(False)

        loss = loss_f.new_zeros(())

        if 'ga' in self.loss_type:
            loss += -loss_f

        elif 'npo' in self.loss_type:
            neg_log_ratio = outputs_f_ref.logits - outputs_f.logits
            loss += -F.logsigmoid(
                self.beta * neg_log_ratio
            ).mean() * 2 / self.beta

        else:
            raise NotImplementedError("Cannot infer the given loss type.")

        if 'gdr' in self.loss_type:
            loss += self.retain_alpha * loss_r

        if 'klf' in self.loss_type:
            raise NotImplementedError("KL forget not implemented yet!")

        if 'klr' in self.loss_type:
            kl_r = F.kl_div(
                outputs_r.logits,
                outputs_r_ref.logits,
                reduction='batchmean',
                log_target=True,
            )
            loss += self.retain_alpha * kl_r

        if self.probe_layers:

            loss_probe_outer = loss.new_zeros(())
            for ell in self.probe_layers:
                probe = self.probes[ell]
                probe.eval()

                # --- FORGET PART ---
                hidden = get_hidden_state_with_grad(
                    model, forget_ids, ell
                )
                lp_f = compute_probe_loss(probe, hidden, forget_labels)

                # --- RETAIN PART ---
                # [r_f_both BLOCK START] - Uncomment to enable retain in probe outer loss
                if self.probe_r_f_both:
                    hidden_r = get_hidden_state_with_grad(
                        model, retain_ids, ell
                    )
                    lp_r = compute_probe_loss(probe, hidden_r, retain_labels)
                    lp = lp_f - lp_r
                else:
                    lp = lp_f
                # [r_f_both BLOCK END]

                loss_probe_outer = loss_probe_outer + lp.to(loss_probe_outer.device)

            loss_probe_outer = loss_probe_outer / len(self.probe_layers)
            loss = loss - self.probe_beta * loss_probe_outer

            for ell in self.probe_layers:
                for p in self.probes[ell].parameters():
                    p.requires_grad_(True)

        return (loss, outputs_f) if return_outputs else loss

    def prediction_step(
        self, model, x, prediction_loss_only: bool, ignore_keys=None
    ):
        input_ids, labels, attention_mask = x
        with torch.no_grad():
            outputs = model(
                input_ids, labels=labels, attention_mask=attention_mask
            )
            logits = outputs.logits
            loss   = outputs.loss
        return (loss, logits, labels)