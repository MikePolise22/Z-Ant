const std = @import("std");
const test_options = @import("test_options");
const test_name = test_options.test_name;

comptime {
    _ = @import("jpeg/test_jpeg_decoder.zig");
    _ = @import("test_utils.zig");
}
