const std = @import("std");
const zant = @import("zant");
const allocator = zant.utils.allocator.allocator;
const IR_zant = @import("../../IR_zant.zig");

// --- onnx ---
const onnx = zant.onnx;
const ModelProto = onnx.ModelProto;
const GraphProto = onnx.GraphProto;
const NodeProto = onnx.NodeProto;
const TensorProto = onnx.TensorProto;

// --- zant IR---
const tensorZant_lib = IR_zant.tensorZant_lib;
const TensorZant = tensorZant_lib.TensorZant;
const TensorCategory = tensorZant_lib.TensorCategory;
const NodeZant_lib = IR_zant.NodeZant_lib;
const NodeZant = NodeZant_lib.NodeZant;
const GraphZant = IR_zant.GraphZant;
const IR_utils = IR_zant.utils;

// --- union ---
const Op_union = @import("../op_union.zig").Op_union;
const operators = IR_zant.operators;

const tensorMath = zant.core.tensor.math_standard;

const utils = IR_zant.utils;

/// Fused Conv+Clip operation for better performance
/// This combines convolution followed by clipping
/// Common pattern in MobileNet v2 and other efficient architectures
pub const Fused_Conv_Clip = struct {
    op_name: []const u8,
    op_Conv: operators.Conv,
    op_Clip: operators.Clip,

    // Cached direct access to frequently used tensors (eliminates indirection)
    input_X: *TensorZant,
    output_Y: *TensorZant,

    pub fn init_fused_op(fusion_list: std.ArrayList(*NodeZant)) !Fused_Conv_Clip {
        // Ensure correct operations are given
        if (fusion_list.items.len != 2) return error.WrongNumberOfElements;
        if (fusion_list.items[0].op != .conv) return error.WrongOpAtPose0;
        if (fusion_list.items[1].op != .clip) return error.WrongOpAtPose2;

        // Extract the specific operations from the unions
        const conv_op = switch (fusion_list.items[0].op) {
            .conv => |c| c,
            else => return error.InvalidConvOperation,
        };

        const clip_op = switch (fusion_list.items[1].op) {
            .clip => |cl| cl,
            else => return error.InvalidClipOperation,
        };

        // Downgrade LINK tensors between fused nodes to FUSED_LINK tensors
        conv_op.output_Y.set_tensorCategory(TensorCategory.FUSED_LINK);

        return Fused_Conv_Clip{
            .op_name = try NodeZant_lib.getFusedOpsName(fusion_list),
            .op_Conv = conv_op,
            .op_Clip = clip_op,
            // OPTIMIZATION: Cache frequently accessed tensors
            .input_X = conv_op.input_X,
            .output_Y = clip_op.output,
        };
    }

    pub fn get_output_shape(self: Fused_Conv_Clip) []usize {
        // OPTIMIZED: Direct access instead of function call
        return self.output_Y.getShape();
    }

    pub fn get_input_tensors(self: Fused_Conv_Clip) ![]*TensorZant {
        // OPTIMIZED: Direct allocation without ArrayList overhead
        const count = 2 + // X, W always present
            (if (self.op_Conv.input_B != null) @as(usize, 1) else 0) +
            (if (self.op_Clip.min != null) @as(usize, 1) else 0) +
            (if (self.op_Clip.max != null) @as(usize, 1) else 0);

        var inputs = try allocator.alloc(*TensorZant, count);
        errdefer allocator.free(inputs);
        var idx: usize = 0;

        // OPTIMIZED: Use cached input_X (direct access)
        inputs[idx] = self.input_X;
        idx += 1;
        inputs[idx] = self.op_Conv.input_W;
        idx += 1;
        if (self.op_Conv.input_B) |b| {
            inputs[idx] = b;
            idx += 1;
        }
        if (self.op_Clip.min) |m| {
            inputs[idx] = m;
            idx += 1;
        }
        if (self.op_Clip.max) |M| {
            inputs[idx] = M;
            idx += 1;
        }

        return inputs;
    }

    pub fn get_output_tensors(self: Fused_Conv_Clip) ![]*TensorZant {
        return self.op_Clip.get_output_tensors();
    }

    pub fn write_op(self: Fused_Conv_Clip, writer: anytype) !void {
        // OPTIMIZATION: Cache ALL sanitized names at once (called only once per name)
        const sanitized_input_name = try utils.getSanitizedName(self.input_X.name);
        const sanitized_kernel_name = try utils.getSanitizedName(self.op_Conv.input_W.name);
        const sanitized_output_name = try utils.getSanitizedName(self.output_Y.name);
        const sanitized_bias_name = if (self.op_Conv.input_B) |b|
            try utils.getSanitizedName(b.name)
        else
            null;

        // OPTIMIZATION: Cache min/max names to avoid repeated getSanitizedName calls
        const sanitized_min_name = if (self.op_Clip.min) |m|
            try utils.getSanitizedName(m.name)
        else
            null;
        const sanitized_max_name = if (self.op_Clip.max) |M|
            try utils.getSanitizedName(M.name)
        else
            null;

        // Build tensor reference strings using cached names
        const tensor_X_string = if (self.input_X.tc == TensorCategory.INITIALIZER)
            try std.fmt.allocPrint(allocator, "@constCast(&param_lib.tensor_{s})", .{sanitized_input_name})
        else
            try std.fmt.allocPrint(allocator, "@constCast(&tensor_{s})", .{sanitized_input_name});
        defer allocator.free(tensor_X_string);

        const tensor_W_string = if (self.op_Conv.input_W.tc == TensorCategory.INITIALIZER)
            try std.fmt.allocPrint(allocator, "@constCast(&param_lib.tensor_{s})", .{sanitized_kernel_name})
        else
            try std.fmt.allocPrint(allocator, "@constCast(&tensor_{s})", .{sanitized_kernel_name});
        defer allocator.free(tensor_W_string);

        const bias_string = if (self.op_Conv.input_B) |_|
            try std.fmt.allocPrint(allocator, "@constCast(&param_lib.tensor_{s})", .{sanitized_bias_name.?})
        else
            try allocator.dupe(u8, "null");
        defer allocator.free(bias_string);

        // Pre-compute casting conditions
        const target_type = self.output_Y.ty.toString();
        const need_kernel_cast = !std.mem.eql(u8, self.op_Conv.input_W.ty.toString(), target_type);
        const need_bias_cast = if (self.op_Conv.input_B) |bias|
            !std.mem.eql(u8, bias.ty.toString(), target_type)
        else
            false;

        // Build stride string
        if (self.op_Conv.strides == null) return error.StrideNotFound;
        const stride_string = try utils.i64SliceToUsizeArrayString(self.op_Conv.strides.?);

        // Build pads string
        const pads_string = if (self.op_Conv.pads) |p| blk: {
            if (p.len > 0) {
                break :blk try utils.i64SliceToUsizeArrayString(p);
            } else {
                break :blk "&[_]usize{}";
            }
        } else "null";

        // Build dilations string
        const dilat_string = if (self.op_Conv.dilations) |d| blk: {
            if (d.len > 0) {
                break :blk try utils.i64SliceToUsizeArrayString(d);
            } else {
                break :blk "&[_]usize{1} ** 2";
            }
        } else "null";

        // OPTIMIZED: Build clip bounds using cached sanitized names
        const min_string = if (self.op_Clip.min) |min_tensor|
            if (min_tensor.tc == TensorCategory.INITIALIZER)
                try std.fmt.allocPrint(allocator, "@constCast(&param_lib.tensor_{s})", .{sanitized_min_name.?})
            else
                try std.fmt.allocPrint(allocator, "&tensor_{s}", .{sanitized_min_name.?})
        else
            "null";
        defer if (!std.mem.eql(u8, min_string, "null")) allocator.free(@constCast(min_string));

        const max_string = if (self.op_Clip.max) |max_tensor|
            if (max_tensor.tc == TensorCategory.INITIALIZER)
                try std.fmt.allocPrint(allocator, "@constCast(&param_lib.tensor_{s})", .{sanitized_max_name.?})
            else
                try std.fmt.allocPrint(allocator, "&tensor_{s}", .{sanitized_max_name.?})
        else
            "null";
        defer if (!std.mem.eql(u8, max_string, "null")) allocator.free(@constCast(max_string));

        // Handle casting with cached sanitized names
        var final_kernel_string = tensor_W_string;
        var final_bias_string = bias_string;
        var need_free_kernel = false;
        var need_free_bias = false;
        defer if (need_free_kernel) allocator.free(@constCast(final_kernel_string));
        defer if (need_free_bias) allocator.free(@constCast(final_bias_string));

        if (need_kernel_cast) {
            final_kernel_string = try std.fmt.allocPrint(allocator, "&tensor_{s}_W_casted_{s}", .{ sanitized_kernel_name, sanitized_output_name });
            need_free_kernel = true;

            _ = try writer.print(
                \\    var tensor_{s}_W_casted_{s} = Tensor({s}).fromShape(&allocator, &param_lib.tensor_{s}.shape) catch return -2;
                \\    tensMath.cast_lean({s}, &param_lib.tensor_{s}, &tensor_{s}_W_casted_{s}) catch return -1;
                \\
            , .{
                sanitized_kernel_name, sanitized_output_name,              target_type,
                sanitized_kernel_name, self.op_Conv.input_W.ty.toString(), sanitized_kernel_name,
                sanitized_kernel_name, sanitized_output_name,
            });
        }

        if (need_bias_cast and self.op_Conv.input_B != null) {
            final_bias_string = try std.fmt.allocPrint(allocator, "&tensor_{s}_B_casted_{s}", .{ sanitized_bias_name.?, sanitized_output_name });
            need_free_bias = true;

            _ = try writer.print(
                \\    var tensor_{s}_B_casted_{s} = Tensor({s}).fromShape(&allocator, &param_lib.tensor_{s}.shape) catch return -2;
                \\    tensMath.cast_lean({s}, &param_lib.tensor_{s}, &tensor_{s}_B_casted_{s}) catch return -1;
                \\
            , .{
                sanitized_bias_name.?, sanitized_output_name,                target_type,
                sanitized_bias_name.?, self.op_Conv.input_B.?.ty.toString(), sanitized_bias_name.?,
                sanitized_bias_name.?, sanitized_output_name,
            });
        }

        // Generate the fused conv+clip call
        _ = try writer.print(
            \\    
            \\    @setEvalBranchQuota(10000);
            \\    // Fused Conv+Clip operation
            \\    tensMath.conv_clip_lean(
            \\        {s},
            \\        {s},
            \\        {s},
            \\        &tensor_{s},
            \\        {s},
            \\        {s},
            \\        {s},
            \\        {s},
            \\        {},
            \\        "{s}",
            \\        {s},
            \\        {s},
            \\    ) catch return -1;
            \\
        , .{
            target_type,
            tensor_X_string,
            final_kernel_string,
            sanitized_output_name,
            final_bias_string,
            stride_string,
            pads_string,
            dilat_string,
            self.op_Conv.group,
            self.op_Conv.auto_pad,
            min_string,
            max_string,
        });
    }

    pub fn compute_output_shape(self: Fused_Conv_Clip) []usize {
        return self.op_Clip.compute_output_shape();
    }

    pub fn print(self: Fused_Conv_Clip) void {
        std.debug.print("\n Fused_Conv_Clip:\n {any}", .{self});
    }

    pub fn sobstitute_tensors(self: *Fused_Conv_Clip, old_tensor: *TensorZant, new_tensor: *TensorZant) !void {
        // OPTIMIZED: Direct pointer comparison without nested try/catch
        if (self.op_Conv.input_X == old_tensor) {
            self.op_Conv.input_X = new_tensor;
            self.input_X = new_tensor; // Keep cache in sync
            return;
        }
        if (self.op_Conv.input_W == old_tensor) {
            self.op_Conv.input_W = new_tensor;
            return;
        }
        if (self.op_Conv.input_B) |b| {
            if (b == old_tensor) {
                self.op_Conv.input_B = new_tensor;
                return;
            }
        }
        if (self.op_Clip.min) |m| {
            if (m == old_tensor) {
                self.op_Clip.min = new_tensor;
                return;
            }
        }
        if (self.op_Clip.max) |M| {
            if (M == old_tensor) {
                self.op_Clip.max = new_tensor;
                return;
            }
        }
        if (self.op_Clip.output == old_tensor) {
            self.op_Clip.output = new_tensor;
            self.output_Y = new_tensor; // Keep cache in sync
            return;
        }

        return error.OldTensorNotFoundInSubstitution;
    }

    // --- Fusion ---
    /// Pattern detection function for Conv -> Clip
    pub fn fn_pattern_detection(graph: *GraphZant, root_node: *NodeZant) anyerror!?std.ArrayList(*NodeZant) {
        _ = graph; // Not used in this sequential pattern

        // Only start detection from Conv nodes
        if (root_node.op != .conv) {
            return null;
        }

        var node_list: std.ArrayList(*NodeZant) = .empty;
        errdefer node_list.deinit(allocator);

        try node_list.append(allocator, root_node);

        if (root_node.next.items.len != 1) {
            node_list.deinit(allocator);
            return null;
        }

        const clip_node = root_node.next.items[0];
        if (clip_node.op != .clip) {
            node_list.deinit(allocator);
            return null;
        }

        try node_list.append(allocator, clip_node);

        std.debug.print(" -> Found complete Conv->Clip pattern!", .{});

        return node_list;
    }

    /// Pattern fusion function
    pub fn fn_pattern_fusion(graph: *GraphZant, node_list: std.ArrayList(*NodeZant)) anyerror!NodeZant {
        _ = graph; // Not used in this sequential pattern

        // Validate the pattern
        if (node_list.items.len != 2) return error.InvalidNumberOfOps;
        if (node_list.items[0].op != .conv) return error.UnexpectedOpAtPos0;
        if (node_list.items[1].op != .clip) return error.UnexpectedOpAtPos1;

        const last_node = node_list.items[1]; // Clip

        // Clone the next list instead of direct reference
        var cloned_next: std.ArrayList(*NodeZant) = .empty;
        for (last_node.next.items) |next_node| {
            try cloned_next.append(allocator, next_node);
        }

        return NodeZant{
            .name = try NodeZant_lib.getFusedOpsName(node_list),
            .op_type = try NodeZant_lib.getFusedOpsType(node_list),
            .op = Op_union{ .fused_Conv_Clip = try init_fused_op(node_list) },
            .next = cloned_next,
            .nodeProto = null,
            .ready = false,
            .is_fused = true,
        };
    }

    /// Pattern substitution function
    pub fn fn_pattern_sobstitution(graph: *GraphZant, fused_node: *NodeZant, node_list: std.ArrayList(*NodeZant)) anyerror!void {
        // Validate inputs
        if (node_list.items.len != 2) return error.InvalidPatternLength;

        const first_node = node_list.items[0]; // Conv node
        const last_node = node_list.items[1]; // Clip node

        // Step 1: Find all predecessor nodes that point to the first node
        const predecessors = try graph.get_predecessors(first_node);

        // Step 2: Update predecessor nodes to point to fused_node
        for (predecessors.items) |predecessor| {
            for (predecessor.next.items, 0..) |next_node, i| {
                if (next_node == first_node) {
                    predecessor.next.items[i] = fused_node;
                }
            }
        }

        // Step 3: Set up fused node's successors
        if (fused_node.next.items.len == 0) {
            for (last_node.next.items) |successor| {
                try fused_node.next.append(allocator, successor);
            }
        }

        // Step 4: Remove old nodes from graph
        try graph.removeNodes(node_list);

        // Step 5: Add fused node to graph
        try graph.nodes.append(allocator, fused_node);
    }
};
