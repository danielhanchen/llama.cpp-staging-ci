// Modifications by the Unsloth AI Inc. team are licensed under AGPL-3.0-only.
#pragma once

#include "common.cuh"

struct rope_corr_dims {
    float v[2];
};


struct mrope_sections {
    int v[4];
};

static __device__ __forceinline__ float rope_yarn_ramp(const float low, const float high, const int i0) {
    const float y = (i0 / 2 - low) / max(0.001f, high - low);
    return 1.0f - min(1.0f, max(0.0f, y));
}

// YaRN algorithm based on LlamaYaRNScaledRotaryEmbedding.py from https://github.com/jquesnelle/yarn
// MIT licensed. Copyright (c) 2023 Jeffrey Quesnelle and Bowen Peng.
template<bool forward>
static __device__ __forceinline__ void rope_yarn(
        const float theta_extrap, const float freq_scale, const rope_corr_dims corr_dims, const int64_t i0, const float ext_factor,
        float mscale, float & cos_theta, float & sin_theta) {
    // Get n-d rotational scaling corrected for extrapolation
    float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    if (ext_factor != 0.0f) {
        float ramp_mix = rope_yarn_ramp(corr_dims.v[0], corr_dims.v[1], i0) * ext_factor;
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;

        // Get n-d magnitude scaling corrected for interpolation
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    cos_theta = cosf(theta) * mscale;
    sin_theta = sinf(theta) * mscale;
    if (!forward) {
        sin_theta *= -1.0f;
    }
}


// the rotated pair of channel i0 (n_offs <= i0 < n_offs + n_dims, i0 even) of one multi-section rope row: xr / dr point at the row
// of x / dst (channel 0), i2 is the token (pos holds 4 sections of ne02 positions). rope_multi uses it, and the rms_norm + rope
// fusion (norm.cu) calls the same code on the normed row, so both give the same floats
template <bool forward, bool has_ff, typename T>
static __device__ __forceinline__ void rope_multi_rotate(const T * xr, T * dr, const int i0, const uint32_t i2, const int ne02,
        const int n_dims, const int n_offs, const int32_t * pos, const float freq_scale, const float ext_factor, const float attn_factor,
        const rope_corr_dims corr_dims, const float theta_scale, const float * freq_factors, const mrope_sections sections,
        const bool is_imrope) {
    const int iw = i0 - n_offs; // relative idx

    const int sect_dims = sections.v[0] + sections.v[1] + sections.v[2] + sections.v[3];
    const int sec_w = sections.v[1] + sections.v[0];
    const int sector = (iw / 2) % sect_dims;

    float theta_base = 0.0;
    if (is_imrope) {
        if (sector % 3 == 1 && sector < 3 * sections.v[1]) {         // h
            theta_base = pos[i2 + ne02 * 1] * powf(theta_scale, iw / 2.0f);
        } else if (sector % 3 == 2 && sector < 3 * sections.v[2]) {  // w
            theta_base = pos[i2 + ne02 * 2] * powf(theta_scale, iw / 2.0f);
        } else if (sector % 3 == 0 && sector < 3 * sections.v[0]) {  // t
            theta_base = pos[i2] * powf(theta_scale, iw / 2.0f);
        } else {
            theta_base = pos[i2 + ne02 * 3] * powf(theta_scale, iw / 2.0f);
        }
    } else {
        if (sector < sections.v[0]) {
            theta_base = pos[i2] * powf(theta_scale, iw / 2.0f);
        } else if (sector >= sections.v[0] && sector < sec_w) {
            theta_base = pos[i2 + ne02 * 1] * powf(theta_scale, iw / 2.0f);
        } else if (sector >= sec_w && sector < sec_w + sections.v[2]) {
            theta_base = pos[i2 + ne02 * 2] * powf(theta_scale, iw / 2.0f);
        } else if (sector >= sec_w + sections.v[2]) {
            theta_base = pos[i2 + ne02 * 3] * powf(theta_scale, iw / 2.0f);
        }
    }

    const float freq_factor = has_ff ? freq_factors[iw/2] : 1.0f;

    float cos_theta;
    float sin_theta;

    rope_yarn<forward>(theta_base/freq_factor, freq_scale, corr_dims, iw, ext_factor, attn_factor, cos_theta, sin_theta);

    // idst/ix point at channel i0/2; the first channel of the rotated pair is n_offs + iw/2 = i0/2 + n_offs/2
    const float x0 = xr[i0/2 + n_offs/2 + 0];
    const float x1 = xr[i0/2 + n_offs/2 + n_dims/2];

    dr[i0/2 + n_offs/2 + 0]        = x0*cos_theta - x1*sin_theta;
    dr[i0/2 + n_offs/2 + n_dims/2] = x0*sin_theta + x1*cos_theta;
}

