"""Reference intermediates for diagnosing the first covariate decode block."""

from dataclasses import dataclass

import torch
from timestral.masks import (
    build_two_stream_positions,
    build_two_stream_time_mask,
    build_two_stream_variate_mask,
    compute_enc_key_validity,
)


@dataclass
class BlockReference:
    tensors: dict[str, torch.Tensor]
    context: torch.Tensor
    observation_mask: torch.Tensor
    location: torch.Tensor
    scale: torch.Tensor


@torch.inference_mode()
def first_block(reference, history, horizon, past, known):
    model = reference.model
    config = model.config
    patch_size = config.patch_size
    inputs, observation_mask = reference._as_context(history, None)
    rows, masks, roles, future, future_mask, _ = reference._assemble_rows(
        inputs, observation_mask, horizon,
        None if past is None else torch.as_tensor(past),
        None if known is None else torch.as_tensor(known),
    )
    row_count = rows.shape[1]
    rows = rows.reshape(row_count, -1)
    masks = masks.reshape(row_count, -1)
    block_patches = min(
        (horizon + patch_size - 1) // patch_size,
        reference.per_shot_horizon_patches,
    )
    steps = block_patches * patch_size
    context_length = rows.shape[1]
    if future is None:
        future = torch.zeros(row_count, steps)
        future_mask = torch.zeros(row_count, steps, dtype=torch.bool)
    else:
        future = future[:, :steps]
        future_mask = future_mask[:, :steps]

    full = torch.cat([rows, future], -1)
    observed = torch.cat([masks, future_mask], -1)
    patches = full.shape[1] // patch_size
    known_rows = roles == 2
    target_rows = roles == 0
    scaled, location, scale = model.scale(full, observed)
    scaled, location, scale = model._known_cov_scaling(
        full, observed, scaled, location, scale, known_rows
    )

    observed_patches = observed.unflatten(-1, (patches, patch_size)).any(-1)
    key_valid = compute_enc_key_validity(observed_patches, config.attend_missing_patches)
    context_patches = torch.full(
        (row_count,), context_length // patch_size, dtype=torch.long
    )
    time_mask = build_two_stream_time_mask(
        context_patches, patches, config.block_size, key_valid, known_rows
    )
    group_ids = torch.zeros(row_count, dtype=torch.long)
    variable_mask = build_two_stream_variate_mask(
        group_ids, key_valid, patches, target_rows, known_rows
    )
    first_observed = observed_patches.any(0).long().argmax().expand(row_count)
    positions = build_two_stream_positions(patches, first_observed)
    features = torch.cat([
        scaled.reshape(row_count, patches, patch_size),
        (~observed).reshape(row_count, patches, patch_size).float(),
    ], -1)
    embedded = model.representation(scaled, observed)
    queries = model.representation.forecast_tokens(row_count, patches)
    normalized = model._run_two_stream(
        embedded, queries, time_mask, variable_mask, positions
    )[:1, context_length:]
    anchor = model.predict_block(
        rows, masks, group_ids, block_patches, roles, future, future_mask
    )[:1]
    refined = reference._refine_block(anchor, rows[:1], masks[:1], None)

    tensors = {
        "scaled": scaled,
        "loc": location,
        "scale": scale,
        "key_valid": key_valid.to(torch.uint8),
        "time_mask": time_mask.to(torch.uint8),
        "variable_mask": variable_mask.to(torch.uint8),
        "positions": positions.float(),
        "patch_features": features,
        "embedded": embedded,
        "normalized_quantiles": normalized,
        "anchor_quantiles": anchor,
        "refined_quantiles": refined,
    }
    return BlockReference(
        tensors=tensors,
        context=rows[:1],
        observation_mask=masks[:1],
        location=location[:1, context_length - 1:context_length, None],
        scale=scale[:1, context_length - 1:context_length, None],
    )
