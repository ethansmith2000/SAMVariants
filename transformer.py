import json
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(torch.nn.Module):
    def __init__(
        self,
        dim,
        heads,
        is_causal=True,
        use_rope=True,
        rope_theta=10000.0,
        qk_norm=True,
    ):
        super().__init__()

        self.dim = dim
        self.heads = heads
        self.is_causal = is_causal
        self.use_rope = use_rope
        self.qk_norm = qk_norm
        self.head_dim = dim // heads
        self.rope_theta = rope_theta

        if self.use_rope and (self.head_dim % 2 != 0):
            raise ValueError("RoPE requires head_dim to be even.")

        self.to_q = torch.nn.Linear(dim, dim, bias=False)
        self.to_k = torch.nn.Linear(dim, dim, bias=False)
        self.to_v = torch.nn.Linear(dim, dim, bias=False)
        self.to_out = torch.nn.Linear(dim, dim, bias=True)

        self.q_norm = torch.nn.RMSNorm(self.head_dim) if qk_norm else torch.nn.Identity()
        self.k_norm = torch.nn.RMSNorm(self.head_dim) if qk_norm else torch.nn.Identity()

    def _apply_rope(self, q, k):
        if not self.use_rope:
            return q, k

        b, h, t, d = q.shape
        half = d // 2
        freqs = torch.arange(half, device=q.device, dtype=q.dtype)
        inv_freq = 1.0 / (self.rope_theta ** (freqs / half))
        positions = torch.arange(t, device=q.device, dtype=q.dtype)
        angles = torch.einsum("t,f->tf", positions, inv_freq)
        sin = angles.sin()[None, None, :, :]
        cos = angles.cos()[None, None, :, :]

        def rotate(x):
            x1, x2 = x[..., :half], x[..., half:]
            return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

        return rotate(q), rotate(k)

    def forward(self, x):
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        q, k, v = map(lambda t: t.view(t.shape[0], t.shape[1], self.heads, -1).transpose(1, 2), (q, k, v))
        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = self._apply_rope(q, k)
        attn = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=self.is_causal)
        # Bring sequence length back to axis 1 before merging heads
        attn = attn.transpose(1, 2).contiguous().view(x.shape[0], x.shape[1], -1)
        return self.to_out(attn)


class FeedForward(torch.nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)

class GeGLU(torch.nn.Module):
    def __init__(self, dim, hidden_dim=None, align_multiple=64):
        super().__init__()
        # Default to GeGLU parity (~2.67x) and align to a friendly GPU multiple
        # if hidden_dim is None:
        hidden_dim = math.ceil(dim * 8 / 3)
        if align_multiple is not None and align_multiple > 1:
            hidden_dim = math.ceil(hidden_dim / align_multiple) * align_multiple
        # First projection produces value and gate in one matmul for efficiency
        self.proj_in = torch.nn.Linear(dim, hidden_dim)
        self.proj_gate = torch.nn.Linear(dim, hidden_dim)
        self.proj_out = torch.nn.Linear(hidden_dim, dim)
        self.act = torch.nn.GELU()

    def forward(self, x):
        value = self.proj_in(x)
        gate = self.proj_gate(x)
        gated = value * self.act(gate)
        return self.proj_out(gated)



class TransformerBlock(torch.nn.Module):
    def __init__(
        self,
        dim,
        heads,
        ff_hidden_dim,
        permute_before_attn=False,
        permute_before_mlp=False,
        permute_kwargs=None,
        is_causal=True,
        use_rope=True,
        rope_theta=10000.0,
        qk_norm=True,
        ffn_type="geglu",
    ):
        super().__init__()
        permute_kwargs = permute_kwargs or {}

        self.attn = Attention(
            dim,
            heads,
            is_causal=is_causal,
            use_rope=use_rope,
            rope_theta=rope_theta,
            qk_norm=qk_norm,
        )
        if ffn_type not in {"geglu", "ffn"}:
            raise ValueError(f"Unsupported ffn_type: {ffn_type}")
        if ffn_type == "geglu":
            self.ff = GeGLU(dim, hidden_dim=ff_hidden_dim)
        else:
            self.ff = FeedForward(dim, hidden_dim=ff_hidden_dim)

        self.norm1 = torch.nn.RMSNorm(dim)
        self.norm2 = torch.nn.RMSNorm(dim)

    def forward(self, x):
        x = self.attn(self.norm1(x)) + x
        x = self.ff(self.norm2(x)) + x
        return x


class Transformer(torch.nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        ff_mult,
        vocab_size,
        max_seq_len,
        gradient_checkpointing=False,
        use_rope=True,
        rope_theta=10000.0,
        qk_norm=True,
        ffn_type="geglu",
    ):
        super().__init__()
        self.token_embedding = torch.nn.Embedding(vocab_size, dim)
        self.use_rope = use_rope
        if not use_rope:
            self.position_embedding = torch.nn.Embedding(max_seq_len, dim)
        self.in_proj = torch.nn.Sequential(
            torch.nn.RMSNorm(dim),
            torch.nn.Linear(dim, dim),
        )
        self.blocks = torch.nn.ModuleList([
            TransformerBlock(
                dim,
                heads,
                dim * ff_mult,
                is_causal=True,
                use_rope=use_rope,
                rope_theta=rope_theta,
                qk_norm=qk_norm,
                ffn_type=ffn_type,  
            ) for i in range(depth)])
        self.config = {
            "dim": dim,
            "depth": depth,
            "heads": heads,
            "ff_mult": ff_mult,
            "vocab_size": vocab_size,
            "max_seq_len": max_seq_len,
            "gradient_checkpointing": gradient_checkpointing,
            "use_rope": use_rope,
            "rope_theta": rope_theta,
            "qk_norm": qk_norm,
            "ffn_type": ffn_type,
        }
        self.out_proj = torch.nn.Sequential(
            torch.nn.RMSNorm(dim),
            torch.nn.Linear(dim, vocab_size),
        )
        self.gradient_checkpointing = gradient_checkpointing

        # Initialize weights following common transformer best practices
        embed_std = dim ** -0.5
        lm_head_std = 0.02

        with torch.no_grad():
            for name, module in self.named_modules():
                if isinstance(module, torch.nn.Embedding):
                    torch.nn.init.normal_(module.weight, mean=0.0, std=embed_std)
                elif isinstance(module, torch.nn.LayerNorm):
                    torch.nn.init.ones_(module.weight)
                    torch.nn.init.zeros_(module.bias)
                elif isinstance(module, torch.nn.RMSNorm):
                    torch.nn.init.ones_(module.weight)
                elif isinstance(module, torch.nn.Linear):
                    # Skip LoRA permuter submodules that already perform their own init
                    if "permute." in name:
                        continue
                    # Attention projections: Q, K, V, O
                    if "attn.to_out" in name:
                        torch.nn.init.xavier_uniform_(module.weight)
                    elif "ff.proj_out" in name:
                        torch.nn.init.xavier_uniform_(module.weight)
                    # Legacy 2-layer MLP init (for older checkpoints/configs)
                    elif "ff.net.0" in name:
                        torch.nn.init.xavier_uniform_(module.weight)
                    elif "ff.net.2" in name:
                        torch.nn.init.xavier_uniform_(module.weight)
                    # Input projection
                    elif "in_proj.1" in name:
                        torch.nn.init.xavier_uniform_(module.weight)
                    # Fallback for any other linear layers
                    else:
                        torch.nn.init.xavier_uniform_(module.weight)

                    if module.bias is not None:
                        torch.nn.init.zeros_(module.bias)

            torch.nn.init.normal_(self.out_proj[1].weight, mean=0.0, std=lm_head_std)
            torch.nn.init.zeros_(self.out_proj[1].bias)

    def forward(self, input_ids, targets=None):
        B, T = input_ids.shape
        pos = torch.arange(0, T, device=input_ids.device)
        tok_emb = self.token_embedding(input_ids)
        if not self.use_rope:
            pos_emb = self.position_embedding(pos)
            x = tok_emb + pos_emb
        else:
            x = tok_emb
        x = self.in_proj(x)
        if self.gradient_checkpointing:
            for block in self.blocks:
                x = torch.utils.checkpoint.checkpoint(block, x, preserve_rng_state=False, use_reentrant=False, determinism_check="none")
        else:
            for block in self.blocks:
                x = block(x)
        logits = self.out_proj(x)
        if targets is not None:
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.shape[-1]), targets.reshape(-1))
            return loss, logits
        else:
            return logits

    def resize_token_embeddings(self, new_size: int):
        if not isinstance(self.token_embedding, torch.nn.Embedding):
            raise NotImplementedError("resize_token_embeddings only supports dense Embedding tables.")

        old_weight = self.token_embedding.weight
        new_emb = torch.nn.Embedding(
            new_size,
            old_weight.shape[1],
            device=old_weight.device,
            dtype=old_weight.dtype,
        )
        with torch.no_grad():
            num_tokens = min(old_weight.shape[0], new_size)
            new_emb.weight[:num_tokens] = old_weight[:num_tokens]
        self.token_embedding = new_emb
        self.config["vocab_size"] = new_size
        return self.token_embedding

    def save_pretrained(self, save_directory, is_main_process=True, save_function=torch.save):
        if not is_main_process:
            return
        os.makedirs(save_directory, exist_ok=True)
        save_function(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(self.config, f, indent=2)
    
