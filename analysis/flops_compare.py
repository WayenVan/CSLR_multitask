#! /usr/bin/env python3


def calculate_self_attention_flops_variable_length(N_q, N_kv, d):
    # Query, Key, Value Projections
    flops_q = N_q * d * d
    flops_k = N_kv * d * d
    flops_v = N_kv * d * d
    flops_projections = flops_q + flops_k + flops_v

    # Q @ K^T: N_q * N_kv * d
    flops_scores = N_q * N_kv * d

    # Softmax: 2 * N_q * N_kv
    flops_softmax = 2 * N_q * N_kv

    # attn @ V: N_q * N_kv * d
    flops_attn_v = N_q * N_kv * d

    # Final projection: N_q * d * d
    flops_output = N_q * d * d

    total_flops = (
        flops_projections + flops_scores + flops_softmax + flops_attn_v + flops_output
    )
    return total_flops


def calculate_multihead_attention_flops_variable_length(N_q, N_kv, d, h):
    d_head = d // h
    # FLOPs per head
    flops_per_head = calculate_self_attention_flops_variable_length(N_q, N_kv, d_head)
    # Total FLOPs for h heads + output projection (N_q * d * d)
    total_flops = h * flops_per_head + N_q * d * d
    return total_flops


def calculate_ffn_flops(N, d, d_ff=None):
    if d_ff is None:
        d_ff = 4 * d  # Default expansion factor (e.g., in original Transformer)

    # First Linear Layer: N * d * d_ff
    flops_linear1 = N * d * d_ff

    # Activation (ReLU/GELU): N * d_ff
    flops_activation = N * d_ff

    # Second Linear Layer: N * d_ff * d
    flops_linear2 = N * d_ff * d

    total_flops = flops_linear1 + flops_activation + flops_linear2
    return total_flops


def calculate_depthwise_conv_flops(H, W, C_in, K, S=1, padding="same"):
    # Compute output dimensions
    if padding == "same":
        H_out = (H + S - 1) // S
        W_out = (W + S - 1) // S
    elif padding == "valid":
        H_out = (H - K + S) // S
        W_out = (W - K + S) // S
    else:
        raise ValueError("Padding must be 'same' or 'valid'")

    # FLOPs = H_out × W_out × K × K × C_in
    flops = H_out * W_out * K * K * C_in
    return flops


def flops_cal(N_q, N_kv, d, h):
    flops_self_attn = (
        calculate_multihead_attention_flops_variable_length(N_q, N_kv, d, h) * 1e-6
    )
    flops_ffn = calculate_ffn_flops(N_q, d) * 1e-6
    flops_conv = calculate_depthwise_conv_flops(1, 1, 1024, 3) * N_q * 1e-6
    print(f"FLOPs self-attn: {flops_self_attn}")
    print(f"FLOPs ffn: {flops_ffn}")
    print(f"FLOPs conv: {flops_conv}")
    return flops_self_attn + flops_ffn + flops_conv


for i in (1, 2, 4, 6):
    print(flops_cal(70, 70 // i, 1024, 8) * 8)
