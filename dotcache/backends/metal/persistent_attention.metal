#include <metal_stdlib>
using namespace metal;

constant uint DIRECT_M0_TILE_D = 32;

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
    const uint token_id = gid.x;
    const uint query_id = gid.y;
    if (query_id >= query_count || token_id >= token_count) {
        return;
    }
    float acc = 0.0f;
    const uint query_base = query_id * padded_head_dim;
    for (uint d = 0; d < padded_head_dim; ++d) {
        acc = fma(
            queries[query_base + d],
            fused_scaled_codes_transposed[d * token_count + token_id],
            acc
        );
    }
    const uint group_base = query_id * num_groups;
    for (uint group_id = 0; group_id < num_groups; ++group_id) {
        acc = fma(
            query_group_sums[group_base + group_id],
            bias[group_id * token_count + token_id],
            acc
        );
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
    const uint token_id = gid.x;
    const uint query_id = gid.y;
    if (query_id >= query_count || token_id >= token_count) {
        return;
    }
    float acc = 0.0f;
    const uint token_offset = token_id * padded_head_dim;
    const uint query_base = query_id * padded_head_dim;
    for (uint d = 0; d < padded_head_dim; ++d) {
        acc = fma(
            queries[query_base + d],
            fused_scaled_codes_flat[token_offset + d],
            acc
        );
    }
    const uint group_base = query_id * num_groups;
    for (uint group_id = 0; group_id < num_groups; ++group_id) {
        acc = fma(
            query_group_sums[group_base + group_id],
            bias[group_id * token_count + token_id],
            acc
        );
    }
    logits[query_id * token_count + token_id] = acc * query_scale;
}

// Tiled transposed-layout direct-M0 score kernel for larger shapes.
// Uses threadgroup staging for a [DIRECT_M0_TILE_D x threadsPerThreadgroup.x] slab.
kernel void direct_m0_logits_transposed_tiled_affine(
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
    uint2 gid                                          [[thread_position_in_grid]],
    uint2 tid                                          [[thread_position_in_threadgroup]],
    uint2 tpg                                          [[threads_per_threadgroup]]
) {
    const uint token_id = gid.x;
    const uint query_id = gid.y;
    if (query_id >= query_count || token_id >= token_count) {
        return;
    }
    const uint tile_token_count = tpg.x;
    const uint tile_query_count = tpg.y;
    threadgroup float query_tile[8 * DIRECT_M0_TILE_D];
    threadgroup float code_tile[DIRECT_M0_TILE_D * 64];
    float acc = 0.0f;
    const uint query_base = query_id * padded_head_dim;

    for (uint dim_base = 0; dim_base < padded_head_dim; dim_base += DIRECT_M0_TILE_D) {
        const uint tile_d = min(DIRECT_M0_TILE_D, padded_head_dim - dim_base);
        const uint linear_tid = tid.y * tile_token_count + tid.x;
        const uint threads_in_group = tile_token_count * tile_query_count;

        const uint query_tile_count = tile_query_count * tile_d;
        for (uint idx = linear_tid; idx < query_tile_count; idx += threads_in_group) {
            const uint local_query_id = idx / tile_d;
            const uint local_dim = idx % tile_d;
            const uint global_query_id = (gid.y - tid.y) + local_query_id;
            query_tile[idx] = global_query_id < query_count
                ? queries[global_query_id * padded_head_dim + dim_base + local_dim]
                : 0.0f;
        }

        const uint code_tile_count = tile_d * tile_token_count;
        for (uint idx = linear_tid; idx < code_tile_count; idx += threads_in_group) {
            const uint local_dim = idx / tile_token_count;
            const uint local_token = idx % tile_token_count;
            const uint global_token_id = (gid.x - tid.x) + local_token;
            code_tile[idx] = global_token_id < token_count
                ? fused_scaled_codes_transposed[(dim_base + local_dim) * token_count + global_token_id]
                : 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint query_tile_base = tid.y * tile_d;
        for (uint local_dim = 0; local_dim < tile_d; ++local_dim) {
            acc = fma(
                query_tile[query_tile_base + local_dim],
                code_tile[local_dim * tile_token_count + tid.x],
                acc
            );
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const uint group_base = query_id * num_groups;
    for (uint group_id = 0; group_id < num_groups; ++group_id) {
        acc = fma(
            query_group_sums[group_base + group_id],
            bias[group_id * token_count + token_id],
            acc
        );
    }
    logits[query_id * token_count + token_id] = acc * query_scale;
}

// Packed group-major 8-bit affine direct-M0 score kernel.
// Layout:
// - payload_words: [num_groups, token_count, words_per_group] flattened
// - scales: [token_count, num_groups]
// - bias: [token_count, num_groups]
// Assumes group_size=32 and bits=8 => words_per_group=8.
kernel void direct_m0_logits_packed_group_major_affine_8bit(
    device const float* queries                 [[buffer(0)]],
    device const float* query_group_sums       [[buffer(1)]],
    device const uint* payload_words           [[buffer(2)]],
    device const float* scales                 [[buffer(3)]],
    device const float* bias                   [[buffer(4)]],
    device float* logits                       [[buffer(5)]],
    constant uint& token_count                 [[buffer(6)]],
    constant uint& query_count                 [[buffer(7)]],
    constant uint& num_groups                  [[buffer(8)]],
    constant uint& words_per_group             [[buffer(9)]],
    constant float& query_scale                [[buffer(10)]],
    uint2 gid                                  [[thread_position_in_grid]]
) {
    const uint token_id = gid.x;
    const uint query_id = gid.y;
    if (query_id >= query_count || token_id >= token_count) {
        return;
    }
    float acc = 0.0f;
    const uint query_group_base = query_id * num_groups;
    const uint query_base = query_id * num_groups * 32;
    for (uint group_id = 0; group_id < num_groups; ++group_id) {
        const uint payload_base = (group_id * token_count + token_id) * words_per_group;
        const uint query_dim_base = query_base + group_id * 32;
        float dot_codes = 0.0f;
        for (uint word_id = 0; word_id < words_per_group; ++word_id) {
            const uint packed = payload_words[payload_base + word_id];
            const float4 qv = float4(
                queries[query_dim_base + word_id * 4 + 0],
                queries[query_dim_base + word_id * 4 + 1],
                queries[query_dim_base + word_id * 4 + 2],
                queries[query_dim_base + word_id * 4 + 3]
            );
            const uchar4 codes = as_type<uchar4>(packed);
            const float4 cv = float4(codes);
            dot_codes = fma(qv.x, cv.x, dot_codes);
            dot_codes = fma(qv.y, cv.y, dot_codes);
            dot_codes = fma(qv.z, cv.z, dot_codes);
            dot_codes = fma(qv.w, cv.w, dot_codes);
        }
        const uint global_scale_bias_index = token_id * num_groups + group_id;
        const float scale_value = scales[global_scale_bias_index];
        const float bias_value = bias[global_scale_bias_index];
        acc = fma(scale_value, dot_codes, acc);
        const float group_sum = query_group_sums[query_group_base + group_id];
        acc = fma(group_sum, bias_value, acc);
    }
    logits[query_id * token_count + token_id] = acc * query_scale;
}
