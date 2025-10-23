const std = @import("std");

test "tests description" {
    std.debug.print("\n--- Running tensor_math tests\n", .{});
}

test {
    _ = @import("test_lib_elementWise_math.zig");
    _ = @import("test_lib_logical_math.zig");
    _ = @import("test_lib_reduction_math.zig");
    _ = @import("test_op_clip.zig");
    _ = @import("test_op_concatenate.zig");
    // _ = @import("test_op_convolution.zig"); Tests are obsolete! The ops are tested into the fuzzing
    _ = @import("test_op_convolution_stm32n6.zig");
    _ = @import("test_op_conv_clip.zig");
    _ = @import("test_op_flatten.zig");
    _ = @import("test_op_gather.zig");
    _ = @import("test_op_gelu.zig");
    _ = @import("test_op_gemm.zig");
    _ = @import("test_op_identity.zig");
    _ = @import("test_op_log.zig");
    _ = @import("test_op_mat_mul.zig");
    _ = @import("test_op_mean.zig");
    _ = @import("test_op_neg.zig");
    _ = @import("test_op_pad.zig");
    // _ = @import("test_op_pooling.zig"); Tests are obsolete! The ops are tested into the fuzzing
    _ = @import("test_op_pow.zig");
    _ = @import("test_op_reshape.zig");
    _ = @import("test_op_resize.zig");
    _ = @import("test_op_shape.zig");
    _ = @import("test_op_split.zig");
    _ = @import("test_op_sqrt.zig");
    _ = @import("test_op_squeeze.zig");
    _ = @import("test_op_transpose.zig");
    _ = @import("test_op_unsqueeze.zig");
    _ = @import("test_other.zig");
}
