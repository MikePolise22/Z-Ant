#include "conv_kernels.h"

#include <limits.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

// Only include math.h and ARM headers when actually compiling for target, not
// during codegen
#ifndef ZANT_CODEGEN_PHASE
#include <math.h>

// Forward declarations for memory functions
extern void *malloc(size_t size);
extern void free(void *ptr);

#if defined(ZANT_HAS_CMSIS_DSP)
#include <arm_math.h>
#if defined(__has_include)
#if __has_include(<arm_nnfunctions.h>)
#define ZANT_HAS_CMSIS_NN 1
#include <arm_nnfunctions.h>
#endif
#endif
#endif
#endif

static size_t g_cmsis_invocations = 0;

extern void zant_stm32n6_reset_ethos_test_state(void);

typedef void (*zant_dot_fn)(const float *, const float *, size_t, float *);

// Forward declarations
static bool cmsis_helium_conv(const float *input, const size_t *input_shape,
                              const float *weights, const size_t *weight_shape,
                              float *output, const size_t *output_shape,
                              const float *bias, size_t bias_len,
                              const size_t *stride, const size_t *pads,
                              const size_t *dilations, size_t group,
                              size_t filters_per_group,
                              size_t channels_per_group);

static bool conv_impl(const float *input, const size_t *input_shape,
                      const float *weights, const size_t *weight_shape,
                      float *output, const size_t *output_shape,
                      const float *bias, size_t bias_len, const size_t *stride,
                      const size_t *pads, const size_t *dilations, size_t group,
                      size_t filters_per_group, size_t channels_per_group,
                      zant_dot_fn dot_fn);

static void reference_dot(const float *a, const float *b, size_t len,
                          float *out) {
  float acc = 0.0f;
  for (size_t i = 0; i < len; ++i) {
    acc += a[i] * b[i];
  }
  *out = acc;
}

#if defined(ZANT_HAS_CMSIS_DSP) && defined(ZANT_HAS_CMSIS_NN) &&               \
    !defined(ZANT_CODEGEN_PHASE)
static inline q7_t quantize_to_q7(float value) {
  if (isnan(value)) {
    return 0;
  }
  if (value > 127.0f) {
    return 127;
  }
  if (value < -128.0f) {
    return -128;
  }
  long rounded = lrintf(value);
  if (rounded > 127) {
    return 127;
  }
  if (rounded < -128) {
    return -128;
  }
  return (q7_t)rounded;
}

static inline int32_t quantize_to_q31(float value) {
  if (isnan(value)) {
    return 0;
  }
  if (value > (float)INT32_MAX) {
    return INT32_MAX;
  }
  if (value < (float)INT32_MIN) {
    return INT32_MIN;
  }
  return (int32_t)lrintf(value);
}

// CMSIS-DSP accelerated dot product function
static void cmsis_dsp_dot(const float *a, const float *b, size_t len,
                          float *out) {
  // Mark that CMSIS path was taken
  zant_stm32n6_mark_cmsis_used();

  // Use CMSIS-DSP arm_dot_prod_f32 for hardware acceleration
  arm_dot_prod_f32(a, b, (uint32_t)len, out);
}

// Move cmsis_helium_conv function implementation inside CMSIS block
static bool cmsis_helium_conv(const float *input, const size_t *input_shape,
                              const float *weights, const size_t *weight_shape,
                              float *output, const size_t *output_shape,
                              const float *bias, size_t bias_len,
                              const size_t *stride, const size_t *pads,
                              const size_t *dilations, size_t group,
                              size_t filters_per_group,
                              size_t channels_per_group) {
  // Use CMSIS-DSP accelerated convolution with arm_dot_prod_f32
  return conv_impl(input, input_shape, weights, weight_shape, output,
                   output_shape, bias, bias_len, stride, pads, dilations, group,
                   filters_per_group, channels_per_group, cmsis_dsp_dot);
}

#elif defined(ZANT_CODEGEN_PHASE)
// Stub types and versions for codegen phase
typedef int8_t q7_t;
typedef int32_t q31_t;

// CMSIS-NN stub types for codegen
typedef struct {
  int32_t *multiplier;
  int32_t *shift;
} cmsis_nn_per_channel_quant_params;

typedef struct {
  int32_t n;
  int32_t h;
  int32_t w;
  int32_t c;
} cmsis_nn_dims;

typedef struct {
  int32_t min;
  int32_t max;
} cmsis_nn_activation;

typedef struct {
  int32_t h;
  int32_t w;
} cmsis_nn_tile;

typedef struct {
  int32_t input_offset;
  int32_t output_offset;
  cmsis_nn_tile stride;
  cmsis_nn_tile padding;
  cmsis_nn_tile dilation;
  cmsis_nn_activation activation;
} cmsis_nn_conv_params;

typedef struct {
  void *buf;
  int32_t size;
} cmsis_nn_context;

typedef enum {
  ARM_CMSIS_NN_SUCCESS = 0,
  ARM_CMSIS_NN_ARG_ERROR = -1
} arm_cmsis_nn_status;

static inline q7_t quantize_to_q7(float value) {
  return (q7_t)(value > 127.0f ? 127 : (value < -128.0f ? -128 : (int)value));
}

static inline int32_t quantize_to_q31(float value) { return (int32_t)value; }

// Stub CMSIS-NN functions for codegen
static inline int32_t
arm_convolve_s8_get_buffer_size(const cmsis_nn_dims *input_dims,
                                const cmsis_nn_dims *filter_dims) {
  (void)input_dims;
  (void)filter_dims;
  return 0;
}

static inline arm_cmsis_nn_status
arm_convolve_s8(const cmsis_nn_context *ctx,
                const cmsis_nn_conv_params *conv_params,
                const cmsis_nn_per_channel_quant_params *quant_params,
                const cmsis_nn_dims *input_dims, const q7_t *input_data,
                const cmsis_nn_dims *filter_dims, const q7_t *filter_data,
                const cmsis_nn_dims *bias_dims, const q31_t *bias_data,
                const cmsis_nn_dims *upscale_dims,
                const cmsis_nn_dims *output_dims, q7_t *output_data) {
  (void)ctx;
  (void)conv_params;
  (void)quant_params;
  (void)input_dims;
  (void)input_data;
  (void)filter_dims;
  (void)filter_data;
  (void)bias_dims;
  (void)bias_data;
  (void)upscale_dims;
  (void)output_dims;
  (void)output_data;
  return ARM_CMSIS_NN_SUCCESS;
}

// Additional stub functions for codegen phase
static inline void *malloc(size_t size) {
  (void)size;
  return NULL;
}
static inline void free(void *ptr) { (void)ptr; }
static inline void *memset(void *s, int c, size_t n) {
  (void)s;
  (void)c;
  (void)n;
  return s;
}

#endif

static bool conv_impl(const float *input, const size_t *input_shape,
                      const float *weights, const size_t *weight_shape,
                      float *output, const size_t *output_shape,
                      const float *bias, size_t bias_len, const size_t *stride,
                      const size_t *pads, const size_t *dilations, size_t group,
                      size_t filters_per_group, size_t channels_per_group,
                      zant_dot_fn dot_fn) {
  if (group == 0 || filters_per_group == 0 || channels_per_group == 0) {
    return false;
  }

  const size_t batch_size = input_shape[0];
  const size_t in_channels = input_shape[1];
  const size_t in_height = input_shape[2];
  const size_t in_width = input_shape[3];

  const size_t out_channels = weight_shape[0];
  const size_t weight_in_channels = weight_shape[1];
  const size_t kernel_height = weight_shape[2];
  const size_t kernel_width = weight_shape[3];

  const size_t out_height = output_shape[2];
  const size_t out_width = output_shape[3];

  if (in_channels != channels_per_group * group) {
    return false;
  }
  if (out_channels != filters_per_group * group) {
    return false;
  }
  if (weight_in_channels != channels_per_group) {
    return false;
  }
  if (bias != NULL && bias_len != out_channels) {
    return false;
  }

  const size_t stride_h = stride[0];
  const size_t stride_w = stride[1];
  const size_t dilation_h = dilations[0];
  const size_t dilation_w = dilations[1];

  const size_t pad_h_begin = pads[0];
  const size_t pad_w_begin = pads[1];

  float *input_patch = NULL;
  float *weight_patch = NULL;
  if (channels_per_group > 0) {
    const size_t bytes = channels_per_group * sizeof(float);
    input_patch = (float *)malloc(bytes);
    weight_patch = (float *)malloc(bytes);
    if (input_patch == NULL || weight_patch == NULL) {
      free(input_patch);
      free(weight_patch);
      return false;
    }
  }

  for (size_t n = 0; n < batch_size; ++n) {
    for (size_t m = 0; m < out_channels; ++m) {
      const size_t current_group = m / filters_per_group;
      const size_t in_channel_start = current_group * channels_per_group;

      const float bias_val = (bias != NULL) ? bias[m] : 0.0f;

      for (size_t oh = 0; oh < out_height; ++oh) {
        for (size_t ow = 0; ow < out_width; ++ow) {
          float sum = bias_val;

          const ptrdiff_t in_h_start =
              (ptrdiff_t)(oh * stride_h) - (ptrdiff_t)pad_h_begin;
          const ptrdiff_t in_w_start =
              (ptrdiff_t)(ow * stride_w) - (ptrdiff_t)pad_w_begin;

          for (size_t kh = 0; kh < kernel_height; ++kh) {
            for (size_t kw = 0; kw < kernel_width; ++kw) {
              const ptrdiff_t in_h = in_h_start + (ptrdiff_t)(kh * dilation_h);
              const ptrdiff_t in_w = in_w_start + (ptrdiff_t)(kw * dilation_w);

              if (in_h >= 0 && in_h < (ptrdiff_t)in_height && in_w >= 0 &&
                  in_w < (ptrdiff_t)in_width) {
                size_t idx = 0;
                for (size_t c = 0; c < channels_per_group; ++c) {
                  const size_t channel = in_channel_start + c;
                  const size_t input_idx =
                      ((n * in_channels + channel) * in_height + (size_t)in_h) *
                          in_width +
                      (size_t)in_w;
                  const size_t weight_idx =
                      ((m * weight_in_channels + c) * kernel_height + kh) *
                          kernel_width +
                      kw;

                  input_patch[idx] = input[input_idx];
                  weight_patch[idx] = weights[weight_idx];
                  idx += 1;
                }

                if (idx > 0) {
                  float dot = 0.0f;
                  dot_fn(input_patch, weight_patch, idx, &dot);
                  sum += dot;
                }
              }
            }
          }

          const size_t output_idx =
              ((n * out_channels + m) * out_height + oh) * out_width + ow;
          output[output_idx] = sum;
        }
      }
    }
  }

  free(input_patch);
  free(weight_patch);

  return true;
}

bool zant_stm32n6_conv_f32(const float *input, const size_t *input_shape,
                           const float *weights, const size_t *weight_shape,
                           float *output, const size_t *output_shape,
                           const float *bias, size_t bias_len,
                           const size_t *stride, const size_t *pads,
                           const size_t *dilations, size_t group,
                           size_t filters_per_group,
                           size_t channels_per_group) {
  return conv_impl(input, input_shape, weights, weight_shape, output,
                   output_shape, bias, bias_len, stride, pads, dilations, group,
                   filters_per_group, channels_per_group, reference_dot);
}

bool zant_stm32n6_conv_f32_helium(const float *input, const size_t *input_shape,
                                  const float *weights,
                                  const size_t *weight_shape, float *output,
                                  const size_t *output_shape, const float *bias,
                                  size_t bias_len, const size_t *stride,
                                  const size_t *pads, const size_t *dilations,
                                  size_t group, size_t filters_per_group,
                                  size_t channels_per_group) {
#if defined(ZANT_HAS_CMSIS_DSP) && defined(ZANT_HAS_CMSIS_NN)
  if (cmsis_helium_conv(input, input_shape, weights, weight_shape, output,
                        output_shape, bias, bias_len, stride, pads, dilations,
                        group, filters_per_group, channels_per_group)) {
    return true;
  }
#endif

  return conv_impl(input, input_shape, weights, weight_shape, output,
                   output_shape, bias, bias_len, stride, pads, dilations, group,
                   filters_per_group, channels_per_group, reference_dot);
}

bool zant_stm32n6_cmsis_s8_selftest(float *output, size_t output_len) {
#if defined(ZANT_HAS_CMSIS_DSP) && defined(ZANT_HAS_CMSIS_NN) &&               \
    !defined(ZANT_CODEGEN_PHASE)
  const size_t expected_count = 4;
  if (output == NULL || output_len < expected_count) {
    return false;
  }

  const q7_t input_data[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  const q7_t weight_data[4] = {1, 0, 0, 1};
  const int32_t bias_data[1] = {0};

  cmsis_nn_dims input_dims = {.n = 1, .h = 3, .w = 3, .c = 1};
  cmsis_nn_dims filter_dims = {.n = 1, .h = 2, .w = 2, .c = 1};
  cmsis_nn_dims bias_dims = {.n = 1, .h = 1, .w = 1, .c = 1};
  cmsis_nn_dims output_dims = {.n = 1, .h = 2, .w = 2, .c = 1};

  cmsis_nn_conv_params conv_params = {
      .input_offset = 0,
      .output_offset = 0,
      .stride = {.h = 1, .w = 1},
      .padding = {.h = 0, .w = 0},
      .dilation = {.h = 1, .w = 1},
      .activation = {.min = -128, .max = 127},
  };

  int32_t multipliers[1] = {128};
  int32_t shifts[1] = {24};
  cmsis_nn_per_channel_quant_params quant_params = {
      .multiplier = multipliers,
      .shift = shifts,
  };

  cmsis_nn_context ctx = {.buf = NULL, .size = 0};
  const int32_t buffer_size =
      arm_convolve_s8_get_buffer_size(&input_dims, &filter_dims);
  if (buffer_size < 0) {
    return false;
  }
  if (buffer_size > 0) {
    ctx.buf = malloc((size_t)buffer_size);
    if (ctx.buf == NULL) {
      return false;
    }
    ctx.size = buffer_size;
  }

  q7_t output_data[4] = {0, 0, 0, 0};
  cmsis_nn_dims upscale_dims = {.n = 1, .h = 1, .w = 1, .c = 1};
  const arm_cmsis_nn_status status =
      arm_convolve_s8(&ctx, &conv_params, &quant_params, &input_dims,
                      input_data, &filter_dims, weight_data, &bias_dims,
                      bias_data, &upscale_dims, &output_dims, output_data);

  if (ctx.buf != NULL) {
    free(ctx.buf);
  }

  if (status != ARM_CMSIS_NN_SUCCESS) {
    return false;
  }

  const float expected[4] = {6.0f, 8.0f, 12.0f, 14.0f};
  for (size_t i = 0; i < expected_count; ++i) {
    if ((float)output_data[i] != expected[i]) {
      return false;
    }
    output[i] = expected[i];
  }

  zant_stm32n6_mark_cmsis_used();
  return true;
#else
  (void)output;
  (void)output_len;
  return false;
#endif
}

void zant_stm32n6_mark_cmsis_used(void) { g_cmsis_invocations += 1; }

void zant_stm32n6_reset_test_state(void) {
  g_cmsis_invocations = 0;
  zant_stm32n6_reset_ethos_test_state();
}

bool zant_stm32n6_cmsis_was_used(void) { return g_cmsis_invocations > 0; }

size_t zant_stm32n6_cmsis_invocation_count(void) { return g_cmsis_invocations; }
