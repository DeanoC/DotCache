#include <metal_stdlib>
using namespace metal;

kernel void persistent_attention_logits(
    device const float* query         [[buffer(0)]],
    device const float* key_cache     [[buffer(1)]],
    device float* logits              [[buffer(2)]],
    constant uint& head_dim           [[buffer(3)]],
    constant uint& token_count        [[buffer(4)]],
    constant float& query_scale       [[buffer(5)]],
    uint gid                          [[thread_position_in_grid]]
) {
    if (gid >= token_count) {
        return;
    }
    float acc = 0.0f;
    const uint token_offset = gid * head_dim;
    for (uint d = 0; d < head_dim; ++d) {
        acc += query[d] * key_cache[token_offset + d];
    }
    logits[gid] = acc * query_scale;
}

kernel void persistent_attention_weighted_sum(
    device const float* weights       [[buffer(0)]],
    device const float* value_cache   [[buffer(1)]],
    device float* output              [[buffer(2)]],
    constant uint& head_dim           [[buffer(3)]],
    constant uint& token_count        [[buffer(4)]],
    uint gid                          [[thread_position_in_grid]]
) {
    if (gid >= head_dim) {
        return;
    }
    float acc = 0.0f;
    for (uint token = 0; token < token_count; ++token) {
        acc += weights[token] * value_cache[token * head_dim + gid];
    }
    output[gid] = acc;
}

// Scaffold for a future large-shape direct-M0 score kernel.
// Layout:
// - queries: [query_count, padded_head_dim]
// - query_group_sums: [query_count, num_groups]
// - fused_scaled_codes_transposed: [padded_head_dim, token_count]
// - bias: [num_groups, token_count]
// - logits: [query_count, token_count]
kernel void direct_m0_logits_transposed_affine(
    device const float* queries                         [[buffer(0)]],
    device const float* query_group_sums               [[buffer(1)]],
    device const float* fused_scaled_codes_transposed  [[buffer(2)]],
    device const float* bias                           [[buffer(3)]],
    device float* logits                               [[buffer(4)]],
    constant uint& padded_head_dim                     [[buffer(5)]],
    constant uint& token_count                         [[buffer(6)]],
    constant uint& query_count                         [[buffer(7)]],
    constant uint& num_groups                          [[buffer(8)]],
    constant float& query_scale                        [[buffer(9)]],
    uint2 gid                                          [[thread_position_in_grid]]
) {
    const uint query_id = gid.x;
    const uint token_id = gid.y;
    if (query_id >= query_count || token_id >= token_count) {
        return;
    }
    float acc = 0.0f;
    for (uint d = 0; d < padded_head_dim; ++d) {
        acc += queries[query_id * padded_head_dim + d] *
               fused_scaled_codes_transposed[d * token_count + token_id];
    }
    const uint groups_per_token_offset = token_id;
    for (uint group_id = 0; group_id < num_groups; ++group_id) {
        acc += query_group_sums[query_id * num_groups + group_id] *
               bias[group_id * token_count + groups_per_token_offset];
    }
    logits[query_id * token_count + token_id] = acc * query_scale;
}

// Flat-layout companion scaffold for smaller direct-M0 shapes.
// Layout:
// - fused_scaled_codes_flat: [token_count, padded_head_dim]
kernel void direct_m0_logits_flat_affine(
    device const float* queries                 [[buffer(0)]],
    device const float* query_group_sums       [[buffer(1)]],
    device const float* fused_scaled_codes_flat[[buffer(2)]],
    device const float* bias                   [[buffer(3)]],
    device float* logits                       [[buffer(4)]],
    constant uint& padded_head_dim             [[buffer(5)]],
    constant uint& token_count                 [[buffer(6)]],
    constant uint& query_count                 [[buffer(7)]],
    constant uint& num_groups                  [[buffer(8)]],
    constant float& query_scale                [[buffer(9)]],
    uint2 gid                                  [[thread_position_in_grid]]
) {
    const uint query_id = gid.x;
    const uint token_id = gid.y;
    if (query_id >= query_count || token_id >= token_count) {
        return;
    }
    float acc = 0.0f;
    const uint token_offset = token_id * padded_head_dim;
    for (uint d = 0; d < padded_head_dim; ++d) {
        acc += queries[query_id * padded_head_dim + d] *
               fused_scaled_codes_flat[token_offset + d];
    }
    for (uint group_id = 0; group_id < num_groups; ++group_id) {
        acc += query_group_sums[query_id * num_groups + group_id] *
               bias[group_id * token_count + token_id];
    }
    logits[query_id * token_count + token_id] = acc * query_scale;
}
