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
