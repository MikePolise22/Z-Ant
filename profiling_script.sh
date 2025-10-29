#!/bin/bash

#################################
# IMPORTANT!! Move me into the root before ./profiling_script
# HOW TO USE:
# - add in models=() what you want to profile, ensure they are "well defined models", see docs/ZANT_CLI.md chapter "1. MOST IMPORTANT - Generate and Test a Model "
# - run : chmod +x profiling_script.sh
# - run : ./profiling_script.sh
#
# It will create a /profiling folder where you can find everything ordered by model with all optimization modes


# Array of models to profile
models=( "mnist-8" ) # "mobilenet_v2" "beer" "fomo8" "coco80_q"

# Array of optimization modes to profile
optimize_modes=( "ReleaseFast" "ReleaseSmall" )

# Create profiling directory if it doesn't exist
mkdir -p profiling

# Function to extract memory leak information from valgrind output
extract_memcheck_info() {
    local output_file="$1"
    echo "=== MEMORY LEAK ANALYSIS ===" >> "$output_file"
    
    # Extract leak summary
    grep -A 10 "LEAK SUMMARY:" valgrind_memcheck.log >> "$output_file" 2>/dev/null || echo "No leak summary found" >> "$output_file"
    
    # Extract heap summary
    grep -A 5 "HEAP SUMMARY:" valgrind_memcheck.log >> "$output_file" 2>/dev/null || echo "No heap summary found" >> "$output_file"
    
    # Extract error summary
    grep "ERROR SUMMARY:" valgrind_memcheck.log >> "$output_file" 2>/dev/null || echo "No error summary found" >> "$output_file"
    
    echo "" >> "$output_file"
}

# Function to extract callgrind information
extract_callgrind_info() {
    local output_file="$1"
    echo "=== CALLGRIND ANALYSIS ===" >> "$output_file"
    
    # Get callgrind output file (most recent)
    local callgrind_file=$(ls -t callgrind.out.* 2>/dev/null | head -n1)
    
    if [ -n "$callgrind_file" ]; then
        echo "Callgrind file: $callgrind_file" >> "$output_file"
        
        # Extract top functions with callgrind_annotate
        callgrind_annotate --threshold=1 "$callgrind_file" 2>/dev/null | head -n 30 >> "$output_file" || echo "Failed to analyze callgrind output" >> "$output_file"
        
        # Move callgrind file to profiling directory
        mv "$callgrind_file" "profiling/"
    else
        echo "No callgrind output file found" >> "$output_file"
    fi
    
    echo "" >> "$output_file"
}

# Function to extract massif information
extract_massif_info() {
    local output_file="$1"
    echo "=== MASSIF ANALYSIS ===" >> "$output_file"
    
    # Get massif output file (most recent)
    local massif_file=$(ls -t massif.out.* 2>/dev/null | head -n1)
    
    if [ -n "$massif_file" ]; then
        echo "Massif file: $massif_file" >> "$output_file"
        
        # Extract heap usage summary
        ms_print "$massif_file" 2>/dev/null | head -n 50 >> "$output_file" || echo "Failed to analyze massif output" >> "$output_file"
        
        # Move massif file to profiling directory
        mv "$massif_file" "profiling/"
    else
        echo "No massif output file found" >> "$output_file"
    fi
    
    echo "" >> "$output_file"
}

# Function to get executable size in a readable format
get_executable_size() {
    if [ -f "./zig-out/bin/main_profiling_target" ]; then
        ls -lh ./zig-out/bin/main_profiling_target | awk '{print $5}'
    else
        echo "N/A (executable not found)"
    fi
}

# Function to profile a model with a specific optimization mode
profile_model_optimization() {
    local model="$1"
    local optimize_mode="$2"
    local report_file="$3"
    local allocation_strategy="$4"

    cg_file="callgrind.out.${model}.${optimize_mode}.$$"
    ms_file="massif.out.${model}.${optimize_mode}.$$"
    
    echo "Profiling $model with $optimize_mode..." | tee -a "$report_file"
    
    echo "=====================================" >> "$report_file"
    echo "PROFILING: $model - $optimize_mode" >> "$report_file"
    echo "Date: $(date)" >> "$report_file"
    echo "=====================================" >> "$report_file"
    echo "" >> "$report_file"
    
    # Build commands
    echo "Building $optimize_mode version..." >> "$report_file"
    zig build lib-gen -Dmodel="$model" -Ddo_export -Doptimize="$optimize_mode" "$allocation_strategy"
    zig build build-main -Dmodel="$model" -Doptimize="$optimize_mode" 
    
    # Check if executable was created
    if [ ! -f "./zig-out/bin/main_profiling_target" ]; then
        echo "ERROR: Executable not found after build for $optimize_mode!" >> "$report_file"
        echo "" >> "$report_file"
        return 1
    fi
    
    # Get executable size
    local exe_size=$(get_executable_size)
    echo "Executable size ($optimize_mode): $exe_size" >> "$report_file"
    echo "" >> "$report_file"
    
    # Clean up old valgrind output files
    rm -f valgrind_*.log callgrind.out.* massif.out.*
    
    # Run Valgrind Memcheck
    echo "Running Valgrind Memcheck for $optimize_mode..." >> "$report_file"

    ./zig-out/bin/main_profiling_target >> "$report_file" 2>&1

    valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes \
             --log-file=valgrind_memcheck.log \
             ./zig-out/bin/main_profiling_target >> "$report_file" 2>&1
    
    extract_memcheck_info "$report_file"
    
    # Run Valgrind Callgrind
    echo "Running Valgrind Callgrind for $optimize_mode..." >> "$report_file"
    valgrind --tool=callgrind --callgrind-out-file="$cg_file" ./zig-out/bin/main_profiling_target >/dev/null 2>&1
    
    extract_callgrind_info "$report_file"
    
    # Run Valgrind Massif
    echo "Running Valgrind Massif for $optimize_mode..." >> "$report_file"
    valgrind --tool=massif --time-unit=ms --massif-out-file="$ms_file" ./zig-out/bin/main_profiling_target >/dev/null 2>&1
    
    extract_massif_info "$report_file"
    
    # Move valgrind log files to profiling directory with optimization mode suffix
    for log_file in valgrind_*.log; do
        if [ -f "$log_file" ]; then
            mv "$log_file" "profiling/${model}_${optimize_mode}_${log_file}" 2>/dev/null || true
        fi
    done
    
    echo "Profiling complete for $model with $optimize_mode" >> "$report_file"
    echo "========================================" >> "$report_file"
    echo "" >> "$report_file"
}

# Function to create performance comparison section
create_performance_comparison() {
    local model="$1"
    local report_file="$2"
    
    echo "=====================================" >> "$report_file"
    echo "PERFORMANCE COMPARISON: $model" >> "$report_file"
    echo "=====================================" >> "$report_file"
    echo "" >> "$report_file"
    
    echo "Summary of optimization modes tested:" >> "$report_file"
    for mode in "${optimize_modes[@]}"; do
        echo "- $mode" >> "$report_file"
    done
    echo "" >> "$report_file"
    
    echo "Key Metrics Comparison:" >> "$report_file"
    echo "Executable Size:" >> "$report_file"
    
    # Create a simple comparison table
    printf "%-15s | %-15s\n" "Optimization" "Executable Size" >> "$report_file"
    printf "%-15s-|-%-15s\n" "---------------" "---------------" >> "$report_file"
    
    # Note: In a real implementation, you would store and compare the actual sizes
    # For now, we'll add placeholders that can be filled manually or by parsing previous output
    for mode in "${optimize_modes[@]}"; do
        printf "%-15s | %-15s\n" "$mode" "See details above" >> "$report_file"
    done
    
    echo "" >> "$report_file"
    echo "Memory Usage and Performance:" >> "$report_file"
    echo "- Check the detailed analysis sections above for each optimization mode" >> "$report_file"
    echo "- Compare heap usage between ReleaseFast and ReleaseSmall" >> "$report_file"
    echo "- ReleaseFast typically offers better runtime performance" >> "$report_file"
    echo "- ReleaseSmall typically produces smaller binaries" >> "$report_file"
    echo "" >> "$report_file"
}

# Function to profile a model with static memory allocation 
profile_model_static() {
    local model="$1"
    local output_dir="profiling/${model}/static"
    
    echo "Profiling $model with all optimization modes..."
    
    # Create output directory
    mkdir -p "$output_dir"
    
    # Single output file for this model with all optimization modes
    local report_file="$output_dir/${model}_complete_profiling_report.txt"
    
    # Clear previous report
    > "$report_file"
    
    echo "=====================================" >> "$report_file"
    echo "COMPLETE PROFILING REPORT: $model" >> "$report_file"
    echo "Date: $(date)" >> "$report_file"
    echo "Optimization modes: ${optimize_modes[*]}" >> "$report_file"
    echo "=====================================" >> "$report_file"
    echo "" >> "$report_file"
    
    # Profile each optimization mode
    for optimize_mode in "${optimize_modes[@]}"; do
        profile_model_optimization "$model" "$optimize_mode" "$report_file" "-Ddynamic=false -Dstatic_planning=true -Duse_tensor_pool=true"
        
        # Add separator between different optimization modes
        echo "" >> "$report_file"
        echo "================================================" >> "$report_file"
        echo "" >> "$report_file"
    done
    
    # Add performance comparison section
    create_performance_comparison "$model" "$report_file"
    
    echo "Complete profiling finished for $model. Results saved in $report_file"
}

# Function to profile a model with dynamic memory allocation 
profile_model_dynamic() {
    local model="$1"
    local output_dir="profiling/${model}/dynamic"
    
    echo "Profiling $model with all optimization modes..."
    
    # Create output directory
    mkdir -p "$output_dir"
    
    # Single output file for this model with all optimization modes
    local report_file="$output_dir/${model}_complete_profiling_report.txt"
    
    # Clear previous report
    > "$report_file"
    
    echo "=====================================" >> "$report_file"
    echo "COMPLETE PROFILING REPORT: $model" >> "$report_file"
    echo "Date: $(date)" >> "$report_file"
    echo "Optimization modes: ${optimize_modes[*]}" >> "$report_file"
    echo "=====================================" >> "$report_file"
    echo "" >> "$report_file"
    
    # Profile each optimization mode
    for optimize_mode in "${optimize_modes[@]}"; do
        profile_model_optimization "$model" "$optimize_mode" "$report_file" "-Ddynamic=true"
        
        # Add separator between different optimization modes
        echo "" >> "$report_file"
        echo "================================================" >> "$report_file"
        echo "" >> "$report_file"
    done
    
    # Add performance comparison section
    create_performance_comparison "$model" "$report_file" 
    
    echo "Complete profiling finished for $model. Results saved in $report_file"
}

# Main execution
echo "Starting comprehensive profiling for all models..."
echo "This may take a while as Valgrind significantly slows execution."
echo "Optimization modes to test: ${optimize_modes[*]}"

for model in "${models[@]}"; do
    echo ""
    echo "Processing model: $model"
    echo "=================================="
    
    # Profile model with all optimization modes
    profile_model_dynamic "$model"
    profile_model_static "$model"
    
done

echo ""
echo "All profiling complete!"
echo "Results are organized in the 'profiling/' directory:"
echo ""
for model in "${models[@]}"; do
    echo "- profiling/${model}_complete_profiling_report.txt (comprehensive report)"
done
echo ""
echo "Each comprehensive report contains:"
echo "  - Detailed analysis for ReleaseFast optimization"
echo "  - Detailed analysis for ReleaseSmall optimization"
echo "  - Performance comparison between optimization modes"
echo "  - Memory usage analysis for each mode"
echo "  - Executable size comparisons"
echo ""
echo "Additional files:"
echo "  - profiling/*_valgrind_memcheck.log (detailed memcheck output by mode)"
echo "  - profiling/callgrind.out.* (callgrind data files)"
echo "  - profiling/massif.out.* (massif data files)"