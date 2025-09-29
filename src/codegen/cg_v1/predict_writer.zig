const std = @import("std");
const zant = @import("zant");
const IR_zant = @import("IR_zant");

// --- zant IR
const GraphZant = IR_zant.GraphZant;
const TensorZant = IR_zant.TensorZant;
const NodeZant = IR_zant.NodeZant;

// --- utils
pub const utils = IR_zant.utils;
// --- onnx
const onnx = zant.onnx;
const ModelOnnx = onnx.ModelProto;
// --- allocator
const allocator = zant.utils.allocator.allocator;

// --- codegen
const codeGenPredict = @import("predict/predict.zig");
const codegen_options = @import("codegen_options");

pub fn write(generated_path: []const u8, model_name: []const u8, linearizedGraph: std.ArrayList(*NodeZant)) !void {

    //initializing writer for lib_operation file
    const lib_file_path = try std.fmt.allocPrint(allocator, "{s}lib_{s}.zig", .{ generated_path, model_name });
    defer allocator.free(lib_file_path);
    var lib_file = try std.fs.cwd().createFile(lib_file_path, .{});
    std.log.info("\n .......... file created, path:{s}", .{lib_file_path});
    defer lib_file.close();

    const writer = lib_file.writer();

    // Write the necessary library imports to the generated Zig file
    try write_libraries(writer);

    // Always write allocation tracking (needed for last_result_size)
    try write_allocationTracking(writer);

    if (codegen_options.log) {
        //log function setting
        try write_logFunction(writer);
    } else {
        // Add dummy log declarations when logging is disabled
        try write_dummyLogFunction(writer);
    }

    //Fixed Buffer Allocator (only for static allocation)
    if (!codegen_options.dynamic) {
        try write_FBA(writer);
    }

    // _ = linearizedGraph;
    // Generate prediction function code
    try codeGenPredict.writePredict(writer, linearizedGraph, codegen_options.do_export);
}

/// Writes the required library imports to the generated Zig file for predict function.
///
/// This function ensures that the necessary standard and package libraries are
/// imported into the generated Zig source file.
///
/// # Parameters
/// - `writer`: A file writer used to write the import statements.
///
/// # Errors
/// This function may return an error if writing to the file fails.
fn write_libraries(writer: std.fs.File.Writer) !void {
    _ = try writer.print(
        \\
        \\ const std = @import("std");
        \\ const zant = @import("zant");
        \\ const Tensor = zant.core.tensor.Tensor;
        \\ const tensMath = zant.core.tensor.math_standard;
        \\ const pkgAllocator = zant.utils.allocator;
        \\ const allocator = pkgAllocator.allocator;
        \\ const utils = @import("codegen").codegen_v1.utils;
        \\ const param_lib = @import("static_parameters.zig");
        \\
    , .{});
}

fn write_allocationTracking(writer: std.fs.File.Writer) !void {
    _ = try writer.print(
        \\
        \\// Global allocation tracking for safe deallocation
        \\var last_result_size: usize = 0;
        \\
        \\// Deallocator function for external C usage
        \\pub {s} fn zant_free_result(ptr: ?[*]T_out) callconv(.C) void {{
        \\    if (ptr) |valid_ptr| {{
        \\        if (last_result_size > 0) {{
        \\            const slice = valid_ptr[0..last_result_size];
        \\            allocator.free(slice);
        \\            last_result_size = 0;
        \\        }}
        \\    }}
        \\}}
        \\
    , .{if (codegen_options.do_export == true) "export" else ""});
}

fn write_logFunction(writer: std.fs.File.Writer) !void {
    _ = try writer.print(
        \\
        \\var log_function: ?*const fn ([*c]const u8) callconv(.C) void = null;
        \\var log_buffer: [512]u8 = undefined;
        \\
        \\fn forward_log(c_msg: [*c]const u8) void {{
        \\    if (log_function) |user_log| {{
        \\        var i: usize = 0;
        \\        while (i < log_buffer.len - 1 and c_msg[i] != 0) : (i += 1) {{
        \\            log_buffer[i] = c_msg[i];
        \\        }}
        \\        log_buffer[i] = 0;
        \\        user_log(@ptrCast(&log_buffer));
        \\    }}
        \\}}
        \\
        \\fn safe_log_forwarder(c_msg: [*c]const u8) callconv(.C) void {{
        \\    forward_log(c_msg);
        \\}}
        \\
        \\fn safe_log_forwarder_qconv(c_msg: [*c]u8) callconv(.C) void {{
        \\    forward_log(@ptrCast(c_msg));
        \\}}
        \\
        \\pub {s} fn setLogFunction(func: ?*const fn ([*c]const u8) callconv(.C) void) void {{
        \\    // Keep only our own logging; disable verbose internal logs
        \\    log_function = func;
        \\    zant.core.tensor.setLogFunction(null);
        \\    if (func != null) {{
        \\        tensMath.setQLinearConvLogFunctionC(@ptrCast(&safe_log_forwarder_qconv));
        \\    }} else {{
        \\        tensMath.setQLinearConvLogFunctionC(null);
        \\    }}
        \\}}
        \\
    , .{if (codegen_options.do_export == true) "export" else ""});
}

fn write_FBA(writer: std.fs.File.Writer) !void {
    //TODO DO AGAIN ALL OF THIS LOGIC it works but it can be way better
    //TODO: instead of hardcoding "buf: [1024 * 10]"" compute the size form the IR Graph

    // Current: 2MB fisso - troppo per modelli piccoli come beer (211KB picco)
    // TODO: Calcolare size dal grafo IR + margine sicurezza 20%
    const buffer_size_kb = if (std.process.getEnvVarOwned(std.heap.page_allocator, "ZANT_FBA_SIZE_KB")) |env_size| blk: {
        defer std.heap.page_allocator.free(env_size);
        break :blk std.fmt.parseInt(u32, env_size, 10) catch 512;
    } else |_| 1024; // Default 1MB to handle peak allocation

    var link_section: ?[]u8 = null;
    if (std.process.getEnvVarOwned(std.heap.page_allocator, "ZANT_FBA_SECTION")) |section_name| {
        link_section = section_name;
    } else |_| {}

    // Use tensor_pool if the option is enabled or if custom section is specified
    const should_use_tensor_pool = codegen_options.use_tensor_pool or link_section != null;

    if (should_use_tensor_pool) {
        const section = link_section orelse ".tensor_pool";
        try writer.print(
            \\
            \\
            \\ // Static allocation: two FixedBufferAllocator pools (ping-pong)
            \\ // Buffer size: {d}KB each (configurable via ZANT_FBA_SIZE_KB env var)
            \\ var buf_a: [{d}]u8 linksection("{s}") = undefined;
            \\ var fba_state_a = std.heap.FixedBufferAllocator.init(&buf_a);
            \\ const fba_a = fba_state_a.allocator();
            \\ var fba_live_a: usize = 0; // live LINK tensors in pool A
            \\
            \\ var buf_b: [{d}]u8 linksection("{s}") = undefined;
            \\ var fba_state_b = std.heap.FixedBufferAllocator.init(&buf_b);
            \\ const fba_b = fba_state_b.allocator();
            \\ var fba_live_b: usize = 0; // live LINK tensors in pool B
            \\ const fba = fba_a; // Backward compatibility path
            \\
            \\
        , .{ buffer_size_kb, buffer_size_kb * 1024, section, buffer_size_kb * 1024, section });
        if (link_section) |section_to_free| {
            std.heap.page_allocator.free(section_to_free);
        }
    } else {
        try writer.print(
            \\
            \\
            \\ // Static allocation: two FixedBufferAllocator pools (ping-pong)
            \\ // Buffer size: {d}KB each (configurable via ZANT_FBA_SIZE_KB env var)
            \\ var buf_a: [{d}]u8 = undefined;
            \\ var fba_state_a = std.heap.FixedBufferAllocator.init(&buf_a);
            \\ const fba_a = fba_state_a.allocator();
            \\ var fba_live_a: usize = 0; // live LINK tensors in pool A
            \\
            \\ var buf_b: [{d}]u8 = undefined;
            \\ var fba_state_b = std.heap.FixedBufferAllocator.init(&buf_b);
            \\ const fba_b = fba_state_b.allocator();
            \\ var fba_live_b: usize = 0; // live LINK tensors in pool B
            \\ const fba = fba_a; // Backward compatibility path
            \\
            \\
        , .{ buffer_size_kb, buffer_size_kb * 1024, buffer_size_kb * 1024 });
    }
}

fn write_dummyLogFunction(writer: std.fs.File.Writer) !void {
    _ = try writer.print(
        \\
        \\// Dummy log declarations for when logging is disabled
        \\var log_function: ?*const fn ([*c]const u8) callconv(.C) void = null;
        \\
        \\inline fn logMsg(comptime msg: []const u8) void {{
        \\    _ = msg; // suppress unused variable warning
        \\}}
        \\
        \\inline fn logf(comptime fmt: []const u8, args: anytype) void {{
        \\    _ = fmt; _ = args; // suppress unused variable warnings
        \\}}
        \\
        \\fn logTensorStatsU8(label: []const u8, t: anytype) void {{
        \\    _ = label; _ = t; // suppress unused variable warnings
        \\}}
        \\
        \\pub {s} fn setLogFunction(func: ?*const fn ([*c]const u8) callconv(.C) void) void {{
        \\    _ = func; // suppress unused variable warning
        \\}}
        \\
    , .{if (codegen_options.do_export == true) "export" else ""});
}
