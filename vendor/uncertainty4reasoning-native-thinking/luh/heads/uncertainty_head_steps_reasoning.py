import torch
import torch.nn as nn
import torch.nn.functional as F

from .uncertainty_head_base import UncertaintyHeadBase

import logging


log = logging.getLogger()


class UncertaintyHeadStepReasoning(UncertaintyHeadBase):
    """
    Step-level uncertainty head optimized for sequential reasoning traces.

    Compared with UncertaintyHeadClaim:
    - runs the transformer encoder ONCE per sequence, not once per claim
    - assumes claims are subsequent contiguous spans
    - marks claim-end / separator positions with a learned embedding
    - predicts uncertainty from the hidden state at each claim end position

    Expected claims format:
        claims is a list of length B
        claims[i] is a tensor of shape [num_claims_i, T]
        each row is a binary mask for one claim span

    Structural assumptions within one sample:
        - each claim is a contiguous non-empty span
        - claims are ordered
        - claims are non-overlapping
        - claims are subsequent:
              start_{k+1} == end_k + 1
    """

    def __init__(
        self,
        feature_extractor,
        head_dim: int = 256,
        n_layers: int = 2,
        n_heads: int = 8,
        dropout: float = 0.1,
        cfg=None,
        mask_future_tokens: bool = False,
    ):
        super().__init__(feature_extractor, cfg=cfg, model_type="step_reasoning")

        self.mask_future_tokens = mask_future_tokens
        self.feature_extractor = feature_extractor

        log.info(f"Feature size: {feature_extractor.feature_dim()}")

        self.proj = nn.Sequential(
            nn.Linear(feature_extractor.feature_dim(), head_dim * 2),
            nn.LayerNorm(head_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_dim * 2, head_dim),
            nn.LayerNorm(head_dim),
            nn.GELU(),
        )

        # 0 = regular token, 1 = claim-end / separator token
        self.step_end_embedding = nn.Embedding(2, head_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=head_dim,
            nhead=n_heads,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            enable_nested_tensor=False,
        )

        self.classifier = nn.Sequential(
            nn.Linear(head_dim, head_dim),
            nn.LayerNorm(head_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(head_dim, 1),
        )

        total_params = sum(p.numel() for p in self.parameters())
        log.info(f"Total number of parameters {total_params}")

    def _extract_span_boundaries(self, entity_mask: torch.Tensor):
        """
        Convert binary claim masks [C, T] into span boundaries.

        Returns:
            starts: LongTensor [C_valid]
            ends: LongTensor [C_valid]
            valid_mask: BoolTensor [C]   # True for non-empty claims
        """
        assert entity_mask.ndim == 2, f"Expected entity_mask [C, T], got {entity_mask.shape}"

        C, T = entity_mask.shape
        if C == 0:
            empty_long = torch.empty(0, dtype=torch.long, device=entity_mask.device)
            empty_bool = torch.empty(0, dtype=torch.bool, device=entity_mask.device)
            return empty_long, empty_long, empty_bool

        if entity_mask.dtype == torch.bool:
            mask = entity_mask
        else:
            unique_vals = torch.unique(entity_mask)
            assert torch.all((unique_vals == 0) | (unique_vals == 1)), (
                f"Claims must be binary 0/1 masks, got values {unique_vals.tolist()}"
            )
            mask = entity_mask.bool()

        token_counts = mask.sum(dim=1)
        valid_mask = token_counts > 0

        # No valid claims at all
        if not torch.any(valid_mask):
            empty_long = torch.empty(0, dtype=torch.long, device=entity_mask.device)
            return empty_long, empty_long, valid_mask

        valid_claims = mask[valid_mask]
        # valid_counts = token_counts[valid_mask]

        starts = torch.argmax(valid_claims.int(), dim=1)

        reversed_mask = torch.flip(valid_claims, dims=[1])
        ends = T - 1 - torch.argmax(reversed_mask.int(), dim=1)

        # expected_lengths = ends - starts + 1
        # assert torch.all(valid_counts == expected_lengths), (
        #     "Each non-empty claim must be a single contiguous span"
        # )
        #
        # if valid_claims.shape[0] > 1:
        #     assert torch.all(starts[1:] > starts[:-1]), (
        #         "Claims must be ordered by start position"
        #     )
        #     assert torch.all(starts[1:] == ends[:-1] + 1), (
        #         "Non-empty claims must be subsequent contiguous spans "
        #         "(each next claim must start exactly after previous ends)"
        #     )

        return starts.long(), ends.long(), valid_mask

    def _build_causal_step_mask(self, T: int, device: torch.device):
        """
        Build a standard causal attention mask:
        token i cannot attend to tokens j > i.

        Returns:
            Bool tensor [T, T], where True means "masked out".
        """
        return torch.triu(
            torch.ones(T, T, device=device, dtype=torch.bool),
            diagonal=1,
        )

    @staticmethod
    def _causal_self_attention(mha, x, dropout_p: float = 0.0):
        """Self-attention with SDPA's own causal masking, no [T, T] tensor.

        Reproduces nn.MultiheadAttention's self-attention path using its packed
        in_proj/out_proj weights. The causal constraint goes to SDPA as a flag,
        which is where a flag is actually honoured -- unlike the encoder's
        `is_causal` argument, which is only a hint about the `mask` it is passed
        alongside.

        dropout_p is the attention-weight dropout nn.MultiheadAttention applies
        internally. SDPA takes it natively, which is what lets this path serve
        training rather than inference only; the caller passes 0.0 in eval.
        """
        batch, length, dim = x.shape
        heads = mha.num_heads
        qkv = F.linear(x, mha.in_proj_weight, mha.in_proj_bias)
        query, key, value = (
            t.view(batch, length, heads, dim // heads).transpose(1, 2)
            for t in qkv.chunk(3, dim=-1)
        )
        attended = F.scaled_dot_product_attention(
            query, key, value, is_causal=True, dropout_p=dropout_p
        )
        return mha.out_proj(attended.transpose(1, 2).reshape(batch, length, dim))

    def _encode_causal_streaming(self, seq_features):
        """The encoder stack, evaluated without materialising T x T anywhere.

        Peak memory for the stock path is quadratic and was measured at 41.7 GB
        for 16k tokens and 93.5 GB for 24k on a 95 GB card, which put a hard
        ceiling just above 24k while the candidate pool's median trace is ~30k
        and its p90 is ~66k. Forcing a different SDPA backend changed nothing --
        the mask tensor keeps every backend on the same path -- so the mask has
        to go, and with it the encoder's own forward.

        Only the layer arrangement is reimplemented; every weight, norm and
        activation comes from the existing module, so this is the same function,
        not a reimplementation of the model. That now includes every dropout the
        stock layer applies -- dropout1 after attention, dropout2 after the feed
        forward block, the FFN's inner dropout, and the attention-weight dropout
        SDPA takes as dropout_p -- so the path is no longer inference-only.

        Post-norm only. This head builds its layers with nn.TransformerEncoder-
        Layer's default, which is post-norm, so a pre-norm arrangement here would
        be a branch nothing ever executed and nothing ever tested -- worse than
        not having it. _can_stream_causal refuses norm_first instead, so such a
        layer falls back to the stock encoder rather than being handed to code
        written for the other arrangement.
        """
        encoder = self.transformer_encoder
        x = seq_features
        for layer in encoder.layers:
            # nn.Dropout consults module.training itself; SDPA takes a float, so
            # only the attention probability has to be switched by hand.
            attn_dropout = layer.self_attn.dropout if self.training else 0.0
            x = layer.norm1(
                x + layer.dropout1(
                    self._causal_self_attention(layer.self_attn, x, attn_dropout)
                )
            )
            x = layer.norm2(
                x + layer.dropout2(
                    layer.linear2(layer.dropout(layer.activation(layer.linear1(x))))
                )
            )
        if encoder.norm is not None:
            x = encoder.norm(x)
        return x

    @staticmethod
    def _is_suffix_padding(seq_padding) -> bool:
        """True when every row's padding is a contiguous run at the end.

        Checked rather than assumed. HFHiddenStateProvider right-pads today, but
        a left-padding provider would make the causal argument below false while
        looking identical here, and the failure would be silent: real positions
        attending to pad, finite logits, no error.
        """
        if seq_padding is None:
            return True
        flags = seq_padding.bool()
        if flags.ndim == 1:
            flags = flags.unsqueeze(0)
        # Non-decreasing along the sequence: once padding starts it never stops.
        return bool((flags[:, 1:].int() - flags[:, :-1].int() >= 0).all())

    def _can_stream_causal(self, seq_padding) -> bool:
        """Whether the streaming path expresses exactly the same computation.

        It now applies every dropout the stock layer does, so training is no
        longer excluded -- which is the point. The stock path materialises a
        [T, T] mask and [heads, T, T] scores, and that cost is what caps
        max_input_length at 32,768 and discards a third of the positive labels
        before training ever sees them.

        Right padding is safe here and does not need the mask: under a causal
        read position i attends only to j <= i, and padding is a suffix, so a
        real position never reaches one. The padded positions themselves compute
        garbage, but the claim masks are zero-filled over them, so nothing
        gathers it. Any other padding layout falls back.

        Two exclusions remain, both for arrangements this head never builds but a
        loaded checkpoint could: a pre-norm layer, which the streaming path does
        not implement, and a MultiheadAttention with bias_k/bias_v/add_zero_attn,
        which it would not reproduce faithfully.
        """
        if not self.mask_future_tokens:
            return False
        if not self._is_suffix_padding(seq_padding):
            return False
        for layer in self.transformer_encoder.layers:
            if getattr(layer, "norm_first", False):
                return False
            mha = layer.self_attn
            if not mha.batch_first or not mha._qkv_same_embed_dim:
                return False
            if mha.bias_k is not None or mha.bias_v is not None or mha.add_zero_attn:
                return False
        return True

    def _compute_tensors(self, llm_inputs, X, X_attn_mask):
        claims = llm_inputs["claims"]

        # [B, T, H_in] -> [B, T, H_head]
        features = self.proj(X)

        # True = padding
        src_key_padding_mask = (X_attn_mask == 0)

        results = []
        batch_size = len(claims)
        seq_len = features.shape[1]

        for i in range(batch_size):
            entity_mask = claims[i]

            assert entity_mask.ndim == 2, (
                f"claims[{i}] must have shape [num_claims_i, T], got {entity_mask.shape}"
            )
            assert entity_mask.shape[1] == seq_len, (
                f"claims[{i}] sequence length {entity_mask.shape[1]} does not match "
                f"feature sequence length {seq_len}"
            )

            num_claims = entity_mask.shape[0]
            if num_claims == 0:
                results.append(features.new_full((0, 1), -100.0))
                continue

            starts, ends, valid_mask = self._extract_span_boundaries(entity_mask)

            # Pre-fill all claims with ignore value
            sample_out = features.new_full((num_claims, 1), -100.0)

            # If all claims are empty, keep ignore-only output
            if valid_mask.sum() == 0:
                results.append(sample_out)
                continue

            # Build one token-level end-marker mask using only valid claims
            end_mask = torch.zeros(seq_len, dtype=torch.long, device=features.device)
            end_mask[ends] = 1

            seq_features = features[i] + self.step_end_embedding(end_mask)
            seq_features = seq_features.unsqueeze(0)
            seq_padding = src_key_padding_mask[i].unsqueeze(0)

            if self._can_stream_causal(seq_padding):
                seq_out = self._encode_causal_streaming(seq_features).squeeze(0)
            else:
                attn_mask = None
                if self.mask_future_tokens:
                    attn_mask = self._build_causal_step_mask(seq_len, seq_features.device)

                # The mask is passed as well as the flag, never instead of it.
                # is_causal is documented as a hint about `mask`, so on its own
                # with mask=None it is silently ignored and attention becomes
                # bidirectional -- every step would see its own future, a
                # semantic change no error would report.
                seq_out = self.transformer_encoder(
                    seq_features,
                    mask=attn_mask,
                    src_key_padding_mask=seq_padding,
                    is_causal=bool(self.mask_future_tokens),
                ).squeeze(0)

            end_representations = seq_out[ends]  # [num_valid_claims, H]
            valid_out = self.classifier(end_representations)  # [num_valid_claims, 1]

            # Put predictions back into original claim slots
            sample_out[valid_mask] = valid_out
            results.append(sample_out)

        max_entities_per_batch = max([o.shape[0] for o in results], default=1)
        padded_results = [
            F.pad(o, (0, 0, 0, max_entities_per_batch - o.shape[0]), value=-100)
            for o in results
        ]

        if len(padded_results) == 0:
            return torch.zeros(0, device=X.device, dtype=features.dtype)

        assert len(results) == batch_size, (
            f"Expected one output tensor per batch item, got {len(results)} for batch size {batch_size}"
        )

        return torch.stack(padded_results, dim=0)

    def forward_from_features(self, features, attention_mask, claims):
        """
        Use precomputed token features directly, bypassing self.feature_extractor.

        Args:
            features: Tensor [B, T, H]
            attention_mask: Tensor [B, T]
            claims: list of length B, each tensor [num_claims_i, T]

        Returns:
            Tensor [B, max_claims_in_batch, 1]
        """
        return self._compute_tensors(
            llm_inputs={"claims": claims},
            X=features,
            X_attn_mask=attention_mask,
        )