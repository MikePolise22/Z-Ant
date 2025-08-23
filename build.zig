const std = @import("std");

/// Entry point for the build system.
/// This function defines how to build the project by specifying various modules and their dependencies.
/// @param b - The build context, which provides utilities for configuring the build process.
pub fn build(b: *std.Build) void {

    // ****************************************************************************************************************
    // ************************************************  BUILD OPTIONS  ***********************************************
    // ****************************************************************************************************************
    const build_options = b.addOptions();
    build_options.addOption(bool, "trace_allocator", b.option(bool, "trace_allocator", "Use a tracing allocator") orelse true);

    // Get target and CPU options from command line or use defaults
    const target_str = b.option([]const u8, "target", "Target architecture (e.g., thumb-freestanding)") orelse "native";
    const cpu_str = b.option([]const u8, "cpu", "CPU model (e.g., cortex_m33)");

    const target_query = std.Target.Query.parse(.{
        .arch_os_abi = target_str,
        .cpu_features = cpu_str,
    }) catch |err| {
        std.log.scoped(.build).warn("Error parsing target: {}\n", .{err});
        return;
    };

    const target = b.resolveTargetQuery(target_query);
    const optimize = b.standardOptimizeOption(.{});

    // Choose appropriate allocator based on target
    const default_allocator = if (target.result.os.tag == .freestanding)
        "ArenaAllocator" // Use ArenaAllocator for freestanding targets (embedded)
    else
        "raw_c_allocator"; // Use raw_c_allocator for hosted targets

    build_options.addOption([]const u8, "allocator", (b.option([]const u8, "allocator", "Allocator to use") orelse default_allocator));

    // ****************************************************************************************************************
    // ************************************************ TESTING OPTIONS ***********************************************
    // ****************************************************************************************************************
    const op_to_test_option = b.option([]const u8, "op", "operator name") orelse "all";

    const testing_options = b.addOptions();
    testing_options.addOption([]const u8, "op", op_to_test_option);

    // ****************************************************************************************************************
    // ************************************************ CODEGEN OPTIONS ***********************************************
    // ****************************************************************************************************************

    // Name and path of the model
    const model_name_option = b.option([]const u8, "model", "Model name") orelse "mnist-8";
    const model_path_option = b.option([]const u8, "model_path", "Model path") orelse std.fmt.allocPrint(b.allocator, "datasets/models/{s}/{s}.onnx", .{ model_name_option, model_name_option }) catch |err| {
        std.log.scoped(.build).warn("Error allocating model path: {}\n", .{err});
        return;
    };
    // Generated path
    var generated_path_option = b.option([]const u8, "generated_path", "Generated path") orelse "";
    if (generated_path_option.len == 0) {
        generated_path_option = std.fmt.allocPrint(b.allocator, "generated/{s}/", .{model_name_option}) catch |err| {
            std.log.scoped(.build).warn("Error allocating generated path: {}\n", .{err});
            return;
        };
    } else {
        if (!std.mem.endsWith(u8, generated_path_option, "/")) {
            generated_path_option = std.fmt.allocPrint(b.allocator, "{s}/", .{generated_path_option}) catch |err| {
                std.log.scoped(.build).warn("Error normalizing path: {}\n", .{err});
                return;
            };
        }
        generated_path_option = std.fmt.allocPrint(b.allocator, "{s}{s}/", .{ generated_path_option, model_name_option }) catch |err| {
            std.log.scoped(.build).warn("Error allocating generated path: {}\n", .{err});
            return;
        };
    }
    const user_tests_option = b.option(bool, "enable_user_tests", "User tests path") orelse false;
    const log_option = b.option(bool, "log", "Run with log") orelse false;
    const shape_option = b.option([]const u8, "shape", "Input shape") orelse "";
    const input_type_option = b.option([]const u8, "type", "Input type") orelse "f32";
    const output_type_option = b.option([]const u8, "output_type", "Output type") orelse "f32";
    const comm_option = b.option(bool, "comm", "Codegen with comments") orelse false;
    const dynamic_option = b.option(bool, "dynamic", "Dynamic allocation") orelse false;
    const export_option = b.option(bool, "do_export", "codegen Exportable ") orelse false;
    const codegen_version_option = b.option([]const u8, "v", "Version, v1 or v2") orelse "v1";
    const xip_option = b.option(bool, "xip", "XIP mode: store weights in flash memory") orelse false;
    const weights_io_option = b.option(bool, "weights_io_mode", "Use configurable weights I/O layer with C callback support") orelse true;

    //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\

    // Define IR codegen options
    const codegen_options = b.addOptions(); // Model name option
    codegen_options.addOption([]const u8, "model", model_name_option);
    codegen_options.addOption([]const u8, "model_path", model_path_option);
    codegen_options.addOption([]const u8, "generated_path", generated_path_option);
    codegen_options.addOption(bool, "user_tests", user_tests_option);
    codegen_options.addOption(bool, "log", log_option);
    codegen_options.addOption(bool, "do_export", export_option);
    codegen_options.addOption([]const u8, "shape", shape_option);
    codegen_options.addOption([]const u8, "type", input_type_option);
    codegen_options.addOption([]const u8, "output_type", output_type_option);
    codegen_options.addOption(bool, "comm", comm_option);
    codegen_options.addOption(bool, "dynamic", dynamic_option);
    codegen_options.addOption([]const u8, "version", codegen_version_option);
    codegen_options.addOption(bool, "xip_mode", xip_option);
    codegen_options.addOption(bool, "weights_io_mode", weights_io_option);

    // --------------------------------------------------------------
    // ---------------------- Modules creation ----------------------
    // --------------------------------------------------------------

    const zant_mod = b.createModule(.{ .root_source_file = b.path("src/zant.zig") });
    zant_mod.addOptions("build_options", build_options);

    const IR_zant_mod = b.createModule(.{ .root_source_file = b.path("src/IR_zant/IR_zant.zig") });
    IR_zant_mod.addImport("zant", zant_mod);
    IR_zant_mod.addOptions("codegen_options", codegen_options);

    const codegen_mod = b.createModule(.{ .root_source_file = b.path("src/codegen/codegen.zig") });
    codegen_mod.addImport("zant", zant_mod);
    codegen_mod.addImport("IR_zant", IR_zant_mod);
    codegen_mod.addOptions("codegen_options", codegen_options);
    IR_zant_mod.addImport("codegen", codegen_mod);

    const Img2Tens_mod = b.createModule(.{ .root_source_file = b.path("src/ImageToTensor/imageToTensor.zig") });
    Img2Tens_mod.addImport("zant", zant_mod);

    //************************************************ UNIT TESTS ************************************************

    // Define unified tests for the project.
    const unit_tests = b.addTest(.{
        .name = "test_lib",
        .root_source_file = b.path("tests/test_lib.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Define test options
    const test_options = b.addOptions();
    test_options.addOption(bool, "heavy", b.option(bool, "heavy", "Run heavy tests") orelse false);
    unit_tests.root_module.addOptions("test_options", test_options);

    const test_name = b.option([]const u8, "test_name", "specify a test name to run") orelse "";
    test_options.addOption([]const u8, "test_name", test_name);

    unit_tests.root_module.addImport("zant", zant_mod);
    unit_tests.root_module.addImport("IR_zant", IR_zant_mod);
    unit_tests.root_module.addImport("codegen", codegen_mod);

    // Only link libc for non-freestanding targets
    if (target.result.os.tag != .freestanding) {
        unit_tests.linkLibC();
    }

    // Add a build step to run all unit tests.
    const run_unit_tests = b.addRunArtifact(unit_tests);
    const test_step = b.step("test", "Run all unit tests");
    test_step.dependOn(&run_unit_tests.step);

    // ************************************************ CODEGEN IR LIB_MODEL************************************************
    //
    // OPTIONS: see codegen_options
    //
    // Define the main executable with target architecture and optimization settings.
    const IR_codeGen_exe = b.addExecutable(.{
        .name = "codegen",
        .root_source_file = b.path("src/codegen/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Only link libc for non-freestanding targets
    if (target.result.os.tag != .freestanding) {
        IR_codeGen_exe.linkLibC();
    }

    // Add necessary imports for the executable.
    IR_codeGen_exe.root_module.addImport("codegen", codegen_mod); //<<-- options are inside this module
    IR_codeGen_exe.root_module.addImport("zant", zant_mod);
    IR_codeGen_exe.root_module.addImport("IR_zant", IR_zant_mod);

    // Define the run command for the main executable.
    const IR_codegen_cmd = b.addRunArtifact(IR_codeGen_exe);
    if (b.args) |args| {
        IR_codegen_cmd.addArgs(args);
    }

    // Create a build step to run the application.
    const IR_codegen_step = b.step("lib-gen", "code generation");
    IR_codegen_step.dependOn(&IR_codegen_cmd.step);

    // ************************************************ LIB_MODEL EXECUTABLE ************************************************
    //
    // OPTIONS: see codegen_options
    //
    // IMPORTANT: for this YOU need to add a main.zig in generated/your_mode/lib_your_mode.zig !!
    //
    const generated_lib_root = std.fmt.allocPrint(b.allocator, "generated/{s}/lib_{s}.zig", .{ model_name_option, model_name_option }) catch |err| {
        std.log.scoped(.build).warn("Error allocating generated_lib_root path: {}\n", .{err});
        return;
    };

    const exe_name = std.fmt.allocPrint(b.allocator, "{s}_exe", .{model_name_option}) catch |err| {
        std.log.scoped(.build).warn("Error allocating exe_name: {}\n", .{err});
        return;
    };

    const lib_model_exe = b.addExecutable(.{
        .name = exe_name,
        .root_source_file = b.path(generated_lib_root),
        .target = target,
        .optimize = optimize,
    });

    // Add necessary imports for the executable.
    lib_model_exe.root_module.addImport("codegen", codegen_mod);
    lib_model_exe.root_module.addImport("zant", zant_mod);

    const model_exe_cmd = b.addRunArtifact(lib_model_exe);
    if (b.args) |args| {
        model_exe_cmd.addArgs(args);
    }

    // Create a build step to run the application.
    const model_exe_step = b.step("lib-exe", "code generation");
    model_exe_step.dependOn(&model_exe_cmd.step);

    // ************************************************ GENERATED LIBRARY TESTS ************************************************

    // Add test for generated library
    const test_model_path = std.fmt.allocPrint(b.allocator, "{s}test_{s}.zig", .{ generated_path_option, model_name_option }) catch |err| {
        std.log.scoped(.build).warn("Error allocating test model path: {}\n", .{err});
        return;
    };

    const test_generated_lib = b.addTest(.{
        .name = "test_generated_lib",
        .root_source_file = b.path(test_model_path),
        .target = target,
        .optimize = .Debug,
    });

    test_generated_lib.root_module.addImport("zant", zant_mod);
    test_generated_lib.root_module.addImport("IR_zant", IR_zant_mod); //codegen
    test_generated_lib.root_module.addImport("codegen", codegen_mod);
    test_generated_lib.linkLibC();

    const run_test_generated_lib = b.addRunArtifact(test_generated_lib);
    const test_step_generated_lib = b.step("lib-test", "Run generated library tests");
    test_step_generated_lib.dependOn(&run_test_generated_lib.step);

    //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//
    ////\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\

    // ************************************************ STATIC LIBRARY CREATION ************************************************

    const lib_model_path = std.fmt.allocPrint(b.allocator, "{s}lib_{s}.zig", .{ generated_path_option, model_name_option }) catch |err| {
        std.log.scoped(.build).warn("Error allocating lib model path: {}\n", .{err});
        return;
    };

    const static_lib = b.addStaticLibrary(.{
        .name = "zant",
        .root_source_file = b.path(lib_model_path),
        .target = target,
        .optimize = .ReleaseSmall,
    });

    // Only link libc for non-freestanding targets
    if (target.result.os.tag != .freestanding) {
        static_lib.linkLibC();
    }

    static_lib.root_module.addImport("zant", zant_mod);
    static_lib.root_module.addImport("codegen", codegen_mod);

    const install_lib_step = b.addInstallArtifact(static_lib, .{ .dest_dir = .{ .override = .{ .custom = model_name_option } } });
    const lib_step = b.step("lib", "Compile tensor_math static library");
    lib_step.dependOn(&install_lib_step.step);

    // Output path for the generated library
    var output_path_option = b.option([]const u8, "output_path", "Output path") orelse "";
    if (output_path_option.len != 0) {
        if (!std.mem.endsWith(u8, output_path_option, "/")) {
            output_path_option = std.fmt.allocPrint(b.allocator, "{s}/", .{output_path_option}) catch |err| {
                std.log.scoped(.build).warn("Error normalizing path: {}\n", .{err});
                return;
            };
        }
        const old_path = std.fmt.allocPrint(b.allocator, "zig-out/{s}/", .{model_name_option}) catch |err| {
            std.log.scoped(.build).warn("Error allocating old path: {}\n", .{err});
            return;
        };
        output_path_option = std.fmt.allocPrint(b.allocator, "{s}{s}/", .{ output_path_option, model_name_option }) catch |err| {
            std.log.scoped(.build).warn("Error allocating output path: {}\n", .{err});
            return;
        };
        const move_step = b.addSystemCommand(&[_][]const u8{
            "mv",
            old_path,
            output_path_option,
        });
        move_step.step.dependOn(&install_lib_step.step);
        lib_step.dependOn(&move_step.step);
        move_step.step.dependOn(&install_lib_step.step);
        lib_step.dependOn(&move_step.step);
    }

    // ************************************************ ONEOP CODEGEN ************************************************
    // Setup oneOp codegen
    // Remember to launch : python3 tests/CodeGen/Python-ONNX/onnx_gen.py to generate the onnx models
    // see: TESTING OPTIONS

    const oneop_codegen_exe = b.addExecutable(.{
        .name = "oneop_codegen",
        .root_source_file = b.path("tests/CodeGen/oneOpModelGenerator.zig"),
        .target = target,
        .optimize = optimize,
    });

    oneop_codegen_exe.root_module.addImport("zant", zant_mod);
    oneop_codegen_exe.root_module.addImport("IR_zant", IR_zant_mod);
    oneop_codegen_exe.root_module.addImport("codegen", codegen_mod); //codegen
    oneop_codegen_exe.root_module.addOptions("testing_options", testing_options); //<<--OSS!! it is an option!
    oneop_codegen_exe.linkLibC();

    const run_oneop_codegen_exe = b.addRunArtifact(oneop_codegen_exe);
    const step_test_oneOp_codegen = b.step("op-codegen-gen", "Codegenerate library tests");
    step_test_oneOp_codegen.dependOn(&run_oneop_codegen_exe.step);

    // ************************************************ ONEOP TESTING ************************************************
    // Setup test_all_oneOp
    // see: TESTING OPTIONS

    const test_all_oneOp = b.addTest(.{
        .name = "test_all_oneOp",
        .root_source_file = b.path("generated/oneOpModels/test_oneop_models.zig"),
        .target = target,
        .optimize = optimize,
    });

    test_all_oneOp.root_module.addImport("zant", zant_mod);
    test_all_oneOp.root_module.addImport("IR_zant", IR_zant_mod);
    test_all_oneOp.root_module.addImport("codegen", codegen_mod); //codegen
    test_all_oneOp.root_module.addOptions("testing_options", testing_options); //<<--OSS!! it is an option!

    test_all_oneOp.linkLibC();

    const run_test_all_oneOp = b.addRunArtifact(test_all_oneOp);

    // ************************************************
    // Setup test oneop
    // It will run
    // - run_oneop_codegen_exe
    // - run_test_all_oneOp

    const step_test_oneOp = b.step("op-codegen-test", "Run generated library tests");
    step_test_oneOp.dependOn(&run_test_all_oneOp.step);

    // ************************************************ NODE EXTRACTOR GEN ************************************************

    const extractor_options = b.addOptions(); // Model name option
    extractor_options.addOption([]const u8, "model", model_name_option);

    const node_extractor_generator = b.addExecutable(.{
        .name = "node_extractor_generator",
        .root_source_file = b.path("tests/CodeGen/node_extractor_generator.zig"),
        .target = target,
        .optimize = optimize,
    });

    node_extractor_generator.root_module.addImport("zant", zant_mod);
    node_extractor_generator.root_module.addImport("IR_zant", IR_zant_mod);
    node_extractor_generator.root_module.addImport("codegen", codegen_mod); //codegen
    node_extractor_generator.root_module.addOptions("extractor_options", extractor_options); //<<--OSS!! it is an option!
    node_extractor_generator.linkLibC();

    const run_node_extractor_generator = b.addRunArtifact(node_extractor_generator);
    const step_node_extractor_generator = b.step("extractor-gen", "Codegenerate tests for extracted nodes ");
    step_node_extractor_generator.dependOn(&run_node_extractor_generator.step);

    // ************************************************ NODE EXTRACTOR TEST ************************************************
    const test_extractor_path = std.fmt.allocPrint(b.allocator, "generated/{s}/extracted/test_extracted_models.zig", .{model_name_option}) catch |err| {
        std.log.scoped(.build).warn("Error allocating test model path: {}\n", .{err});
        return;
    };

    const test_node_extractor = b.addTest(.{
        .name = "test_node_extractor",
        .root_source_file = b.path(test_extractor_path),
        .target = target,
        .optimize = optimize,
    });

    test_node_extractor.root_module.addImport("zant", zant_mod);
    test_node_extractor.root_module.addImport("IR_zant", IR_zant_mod);
    test_node_extractor.root_module.addImport("codegen", codegen_mod); //codegen
    test_node_extractor.root_module.addOptions("extractor_options", extractor_options); //<<--OSS!! it is an option!
    test_node_extractor.linkLibC();

    const run_test_extractor = b.addRunArtifact(test_node_extractor);
    const step_test_extractor = b.step("extractor-test", "Start extracted nodes tests");
    step_test_extractor.dependOn(&run_test_extractor.step);

    // ************************************************ BENCHMARK  ************************************************

    const benchmark = b.addExecutable(.{
        .name = "benchmark",
        .root_source_file = b.path("benchmarks/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    const bench_options = b.addOptions();
    bench_options.addOption(bool, "full", b.option(bool, "full", "Choose whenever run full benchmark or not") orelse false);

    benchmark.root_module.addImport("zant", zant_mod);
    benchmark.root_module.addImport("codegen", codegen_mod); //codegen
    benchmark.root_module.addOptions("bench_options", bench_options);
    benchmark.linkLibC();

    const run_benchmark = b.addRunArtifact(benchmark);
    const benchmark_step = b.step("benchmark", "Run benchmarks");
    benchmark_step.dependOn(&run_benchmark.step);

    // ************************************************ ONNX PARSER TESTS ************************************************
    // Add test for generated library

    const test_onnx_parser = b.addTest(.{
        .name = "test_generated_lib",
        .root_source_file = b.path("tests/Onnx/onnx_loader.zig"),
        .target = target,
        .optimize = optimize,
    });

    test_onnx_parser.root_module.addImport("zant", zant_mod);
    test_onnx_parser.linkLibC();

    const run_test_onnx_parser = b.addRunArtifact(test_onnx_parser);
    const step_test_onnx_parser = b.step("onnx-parser", "Run generated library tests");
    step_test_onnx_parser.dependOn(&run_test_onnx_parser.step);

    // ************************************************ MAIN EXECUTABLE (for profiling) ************************************************

    // Path to the generated model options file (moved here)
    const model_options_path = std.fmt.allocPrint(b.allocator, "{s}model_options.zig", .{generated_path_option}) catch |err| {
        std.log.scoped(.build).warn("Error allocating model options path: {}\n", .{err});
        return;
    };

    const main_executable = b.addExecutable(.{
        .name = "main_profiling_target",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    main_executable.linkLibC();
    main_executable.linkLibrary(static_lib);
    const model_opts_mod = b.createModule(.{
        .root_source_file = b.path(model_options_path),
    });
    model_opts_mod.addImport("zant", zant_mod);
    model_opts_mod.addImport("IR_zant", IR_zant_mod);
    main_executable.root_module.addImport("model_opts", model_opts_mod);
    const install_main_exe_step = b.addInstallArtifact(main_executable, .{}); // Installa l'eseguibile

    const build_main_step = b.step("build-main", "Build the main executable for profiling");
    build_main_step.dependOn(&install_main_exe_step.step);
}
