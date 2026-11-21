#pragma OPENCL EXTENSION cl_khr_fp16 : enable

//------------------------------------------------------------------------------
// softplus
//------------------------------------------------------------------------------

// log(1 + e) as before; only where 1 + e rounds to 1 (e <= 2^-24, x < -16.6, where it returned 0) it returns e = log1p(e) to F32 precision.
// The test is on e, not on 1 + e == 1, which fast-math compilers may fold to e == 0
inline float softplus_tail(float x) {
    if (x > 20.0f) {
        return x;
    }
    const float e = exp(x);
    return (e <= 5.9604644775390625e-8f) ? e : log(1.0f + e);
}

inline float4 softplus_tail4(float4 x) {
    const float4 e = exp(x);
    return select(select(log(1.0f + e), e, e <= 5.9604644775390625e-8f), x, x > 20.0f);
}

kernel void kernel_softplus_f32(
        global const float * src0,
        ulong                offset0,
        global       float * dst,
        ulong                offsetd
) {
    src0 = (global float*)((global char*)src0 + offset0);
    dst  = (global float*)((global char*)dst + offsetd);

    dst[get_global_id(0)] = softplus_tail(src0[get_global_id(0)]);
}

kernel void kernel_softplus_f32_4(
        global const float4 * src0,
        ulong                 offset0,
        global       float4 * dst,
        ulong                 offsetd
) {
    src0 = (global float4*)((global char*)src0 + offset0);
    dst  = (global float4*)((global char*)dst + offsetd);

    dst[get_global_id(0)] = softplus_tail4(src0[get_global_id(0)]);
}

kernel void kernel_softplus_f16(
        global const half * src0,
        ulong               offset0,
        global       half * dst,
        ulong               offsetd
) {
    src0 = (global half*)((global char*)src0 + offset0);
    dst  = (global half*)((global char*)dst + offsetd);

    const float x = convert_float(src0[get_global_id(0)]);
    dst[get_global_id(0)] = convert_half_rte(softplus_tail(x));
}

kernel void kernel_softplus_f16_4(
        global const half4 * src0,
        ulong                offset0,
        global       half4 * dst,
        ulong                offsetd
) {
    src0 = (global half4*)((global char*)src0 + offset0);
    dst  = (global half4*)((global char*)dst + offsetd);

    const float4 x = convert_float4(src0[get_global_id(0)]);
    dst[get_global_id(0)] = convert_half4_rte(softplus_tail4(x));
}

kernel void kernel_softplus_f32_nc(
        global const char * src0,
        ulong               offset0,
        global       char * dst,
        ulong               offsetd,
        int   ne00,
        ulong nb00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        ulong nb0,
        ulong nb1,
        ulong nb2,
        ulong nb3
) {
    src0 = src0 + offset0;
    dst  = dst + offsetd;

    const int i3 = get_group_id(2);
    const int i2 = get_group_id(1);
    const int i1 = get_group_id(0);

    for (int i0 = get_local_id(0); i0 < ne00; i0 += get_local_size(0)) {
        global const float * x = (global const float *)(src0 + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
        global       float * y = (global       float *)(dst  + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);

        *y = softplus_tail(*x);
    }
}

kernel void kernel_softplus_f16_nc(
        global const char * src0,
        ulong               offset0,
        global       char * dst,
        ulong               offsetd,
        int   ne00,
        ulong nb00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        ulong nb0,
        ulong nb1,
        ulong nb2,
        ulong nb3
) {
    src0 = src0 + offset0;
    dst  = dst + offsetd;

    const int i3 = get_group_id(2);
    const int i2 = get_group_id(1);
    const int i1 = get_group_id(0);

    for (int i0 = get_local_id(0); i0 < ne00; i0 += get_local_size(0)) {
        global const half * hx = (global const half *)(src0 + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
        global       half * hy = (global       half *)(dst  + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);

        const float x = convert_float(*hx);
        *hy = convert_half_rte(softplus_tail(x));
    }
}
