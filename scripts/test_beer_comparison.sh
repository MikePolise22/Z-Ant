#!/bin/bash

# Beer Performance Comparison Script
# Builds two different beer libraries and tests them independently

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="$REPO_ROOT/build/beer_comparison"
RESULTS_FILE="$BUILD_DIR/comparison_results.txt"

echo "=== Beer Performance Comparison ==="
echo "Repository: $REPO_ROOT"
echo "Build directory: $BUILD_DIR"
echo

# Create build directory
mkdir -p "$BUILD_DIR"
rm -f "$RESULTS_FILE"

cd "$REPO_ROOT"

echo "=== Step 1: Generate and Build Beer Library WITHOUT CMSIS-NN ==="
echo "Generating beer model with tensor pool support..."

# Generate beer model with tensor pool allocation
zig build lib-gen -Dmodel=beer -Dfuse -Duse_tensor_pool=true -Ddo_export=true

echo "Building reference beer library (no CMSIS acceleration)..."

# Clean previous builds  
rm -rf zig-out/beer/

# Build beer library without CMSIS-NN acceleration
zig build lib \
    -Dmodel=beer \
    -Dfuse \
    -Duse_tensor_pool=true \
    -Ddo_export=true \
    -Doutput_path="$BUILD_DIR" \
    -Dtarget=thumb-freestanding \
    -Dcpu=cortex_m55 \
    -Doptimize=ReleaseFast

# Backup the reference library (support both output_path and default zig-out path)
if [ -f "$BUILD_DIR/libzant.a" ]; then
    cp "$BUILD_DIR/libzant.a" "$BUILD_DIR/libzant_reference.a"
elif [ -f "zig-out/beer/libzant.a" ]; then
    cp "zig-out/beer/libzant.a" "$BUILD_DIR/libzant_reference.a"
else
    echo "error: libzant.a not found after build (checked $BUILD_DIR and zig-out/beer)" >&2
    exit 1
fi
echo "✅ Reference library built and backed up"

echo
echo "=== Step 2: Test Reference (No CMSIS-NN) ==="

# Create a simple test script that uses the existing test infrastructure
cat > "$BUILD_DIR/test_reference.py" << 'EOF'
#!/usr/bin/env python3
import sys
import os
import subprocess
import shutil
from pathlib import Path

# Add the scripts directory to path to import the original test module
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

# Import the original test function components
from test_stm32n6_qemu import run_qemu, detect_qemu, ArmGccToolchain

REPO_ROOT = Path(__file__).parent.parent.parent
BUILD_DIR = Path(__file__).parent
HARNESS_DIR = REPO_ROOT / "tests/stm32n6_qemu"
GENERATED_DIR = REPO_ROOT / "generated/beer"
STM32_DIR = REPO_ROOT / "src/Core/Tensor/Accelerators/stm32n6"

# Copy back the reference library
ref_lib = BUILD_DIR / "libzant_reference.a"
target_lib = REPO_ROOT / "zig-out/beer/libzant.a"
target_lib.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(ref_lib, target_lib)

# Use the exact same build process as the original script
BEER_SOURCES = (
    HARNESS_DIR / "runtime.c",
    HARNESS_DIR / "semihost_arm.c",
    HARNESS_DIR / "semihost_arm.S",
    HARNESS_DIR / "common" / "support.c",
    HARNESS_DIR / "beer_main.c",
    STM32_DIR / "conv_f32.c",
    STM32_DIR / "ethos_stub.c",
)

toolchain = ArmGccToolchain("arm-none-eabi")
qemu_path = detect_qemu(None)

elf_path = BUILD_DIR / "beer_reference.elf"

# Build reference executable using original build process
heap_kb = int(os.environ.get("ZANT_HEAP_KB", "2048"))
toolchain.build(
    output=elf_path,
    base_sources=BEER_SOURCES,
    macros=(*toolchain.default_macros(), f"ZANT_HEAP_KB={heap_kb}"),

    extra_sources=(target_lib,),  # Only the beer library
    include_dirs=(GENERATED_DIR,),
)

print("✅ Reference executable built using original build process")

import re
cycle_re = re.compile(r"beer PASS .*?cycles=(\d+) .*?time_ms=([0-9.]+) fps=([0-9.]+)")

# Run test using FVP by default if detected (ZANT_USE_FVP=1 default)
print("Running reference test (FVP preferred)...")
import time
start_time = time.perf_counter()
from test_stm32n6_qemu import detect_fvp, run_fvp
fvp_path = detect_fvp(os.environ.get("FVP_BIN"))
if fvp_path and (os.environ.get("ZANT_USE_FVP", "1") != "0"):
    result = run_fvp(
        fvp_path,
        elf_path,
        verbose=False,
        success_marker="beer PASS",
        timeout=30.0
    )
    if result.returncode != 0:
        print("FVP failed to run, falling back to QEMU...")
        result = run_qemu(
            qemu_path,
            elf_path,
            verbose=False,
            success_marker="beer PASS",
            timeout=30.0
        )
else:
    result = run_qemu(
        qemu_path,
        elf_path,
        verbose=False,
        success_marker="beer PASS",
        timeout=30.0
    )
end_time = time.perf_counter()

if result.returncode == 0:
    duration = (end_time - start_time) * 1000
    m = cycle_re.search(result.stdout)
    if m:
        # Prefer device-time from cycles if available and non-zero; otherwise fallback to host time
        ref_ms = float(m.group(2))
        if ref_ms <= 0.0:
            ref_ms = duration
            print(f"✅ Reference test PASSED host_time={ref_ms:.2f} ms (no device cycles)")
        else:
            ref_fps = float(m.group(3)) if len(m.groups()) >= 3 else (1000.0 / ref_ms if ref_ms > 0 else 0.0)
            print(f"✅ Reference test PASSED device_time={ref_ms:.2f} ms ({ref_fps:.2f} fps) (host {duration:.2f} ms)")
    else:
        ref_ms = duration
        print(f"✅ Reference test PASSED host_time={ref_ms:.2f} ms")
    with open(BUILD_DIR / "comparison_results.txt", "w") as f:
        f.write(f"beer_reference: {ref_ms:.2f} ms\n")
else:
    print("❌ Reference test FAILED")
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    sys.exit(1)
EOF

python3 "$BUILD_DIR/test_reference.py"

echo
echo "=== Step 3: Generate and Build Beer Library WITH CMSIS-NN ==="
echo "Regenerating beer model for CMSIS-NN build..."

# Regenerate beer model (same generation works for both)
zig build lib-gen -Dmodel=beer -Dfuse -Duse_tensor_pool=true -Ddo_export=true

echo "Building optimized beer library (with CMSIS-NN acceleration)..."

# Clean previous builds
rm -rf zig-out/beer/

# Build beer library with CMSIS-NN acceleration
zig build lib \
    -Dmodel=beer \
    -Dfuse \
    -Duse_tensor_pool=true \
    -Ddo_export=true \
    -Dstm32n6_accel=true \
    -Dstm32n6_use_cmsis=true \
    -Dstm32n6_use_ethos=true \
    -Doutput_path="$BUILD_DIR" \
    -Dtarget=thumb-freestanding \
    -Dcpu=cortex_m55 \
    -Doptimize=ReleaseFast

# Backup the CMSIS library (support both output_path and default zig-out path)
if [ -f "$BUILD_DIR/libzant.a" ]; then
    cp "$BUILD_DIR/libzant.a" "$BUILD_DIR/libzant_cmsis.a"
elif [ -f "zig-out/beer/libzant.a" ]; then
    cp "zig-out/beer/libzant.a" "$BUILD_DIR/libzant_cmsis.a"
else
    echo "error: CMSIS libzant.a not found after build (checked $BUILD_DIR and zig-out/beer)" >&2
    exit 1
fi
echo "✅ CMSIS-NN library built and backed up"

echo
echo "=== Step 4: Test CMSIS-NN Optimized ==="

# Create test script for CMSIS-NN version
cat > "$BUILD_DIR/test_cmsis.py" << 'EOF'
#!/usr/bin/env python3
import sys
import os
import subprocess
import shutil
from pathlib import Path

# Add the scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from test_stm32n6_qemu import run_qemu, detect_qemu, ArmGccToolchain, get_cmsis_sources, get_cmsis_dsp_sources

REPO_ROOT = Path(__file__).parent.parent.parent
BUILD_DIR = Path(__file__).parent
HARNESS_DIR = REPO_ROOT / "tests/stm32n6_qemu"
GENERATED_DIR = REPO_ROOT / "generated/beer"
STM32_DIR = REPO_ROOT / "src/Core/Tensor/Accelerators/stm32n6"

# Copy back the CMSIS library
cmsis_lib = BUILD_DIR / "libzant_cmsis.a"
target_lib = REPO_ROOT / "zig-out/beer/libzant.a"
target_lib.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(cmsis_lib, target_lib)

# Get CMSIS sources
nn_include = REPO_ROOT / "third_party/CMSIS-NN/Include"
dsp_include = REPO_ROOT / "third_party/CMSIS-DSP/Include"
convolve_source = REPO_ROOT / "third_party/CMSIS-NN/Source/ConvolutionFunctions/arm_convolve_s8.c"

cmsis_nn_sources = get_cmsis_sources(convolve_source, nn_include)
cmsis_dsp_sources = get_cmsis_dsp_sources(dsp_include)

BEER_SOURCES = (
    HARNESS_DIR / "runtime.c",
    HARNESS_DIR / "semihost_arm.c",
    HARNESS_DIR / "semihost_arm.S",
    HARNESS_DIR / "common" / "support.c",
    HARNESS_DIR / "beer_main.c",
    STM32_DIR / "conv_f32.c",
    STM32_DIR / "ethos_stub.c",
)

# Setup includes
include_dirs = [dsp_include]
if nn_include != dsp_include:
    include_dirs.append(nn_include)
cmsis_core = REPO_ROOT / "third_party/CMSIS_5/CMSIS/Core/Include"
if cmsis_core.exists():
    include_dirs.append(cmsis_core)
include_dirs.append(GENERATED_DIR)

toolchain = ArmGccToolchain("arm-none-eabi")
qemu_path = detect_qemu(None)

elf_path = BUILD_DIR / "beer_cmsis.elf"

# Build CMSIS executable using original build process
heap_kb = int(os.environ.get("ZANT_HEAP_KB", "2048"))
toolchain.build(
    output=elf_path,
    base_sources=BEER_SOURCES,
    macros=(*toolchain.default_macros(), "ZANT_HAS_CMSIS_DSP=1", "ZANT_HAS_CMSIS_NN=1", f"ZANT_HEAP_KB={heap_kb}"),
    extra_sources=(*cmsis_nn_sources, *cmsis_dsp_sources, target_lib),
    include_dirs=tuple(include_dirs),
)

print("✅ CMSIS-NN executable built using original build process")

print("Running CMSIS-NN optimized test (FVP preferred)...")
import time
start_time = time.perf_counter()
from test_stm32n6_qemu import detect_fvp, run_fvp
fvp_path = detect_fvp(os.environ.get("FVP_BIN"))
if fvp_path and (os.environ.get("ZANT_USE_FVP", "1") != "0"):
    result = run_fvp(
        fvp_path,
        elf_path,
        verbose=False,
        success_marker="beer PASS",
        timeout=30.0
    )
    if result.returncode != 0:
        print("FVP failed to run, falling back to QEMU...")
        result = run_qemu(
            qemu_path,
            elf_path,
            verbose=False,
            success_marker="beer PASS",
            timeout=30.0
        )
else:
    result = run_qemu(
        qemu_path,
        elf_path,
        verbose=False,
        success_marker="beer PASS",
        timeout=30.0
    )
end_time = time.perf_counter()

if result.returncode == 0:
    duration = (end_time - start_time) * 1000
    import re
    cycle_re = re.compile(r"beer PASS .*?cycles=(\d+) .*?time_ms=([0-9.]+) fps=([0-9.]+)")
    m = cycle_re.search(result.stdout)
    if m:
        cmsis_ms = float(m.group(2))
        if cmsis_ms <= 0.0:
            cmsis_ms = duration
            print(f"✅ CMSIS-NN test PASSED host_time={cmsis_ms:.2f} ms (no device cycles)")
        else:
            cmsis_fps = float(m.group(3)) if len(m.groups()) >= 3 else (1000.0 / cmsis_ms if cmsis_ms > 0 else 0.0)
            print(f"✅ CMSIS-NN test PASSED device_time={cmsis_ms:.2f} ms ({cmsis_fps:.2f} fps) (host {duration:.2f} ms)")
    else:
        cmsis_ms = duration
        print(f"✅ CMSIS-NN test PASSED host_time={cmsis_ms:.2f} ms")
    with open(BUILD_DIR / "comparison_results.txt", "a") as f:
        f.write(f"beer_cmsis_nn: {cmsis_ms:.2f} ms\n")
else:
    print("❌ CMSIS-NN test FAILED")
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    sys.exit(1)
EOF

python3 "$BUILD_DIR/test_cmsis.py"

echo
echo "=== Results Summary ==="
if [ -f "$RESULTS_FILE" ]; then
    cat "$RESULTS_FILE"
    echo
    
    # Calculate performance improvement
    python3 - << EOF
import re
from pathlib import Path

results_file = Path("$RESULTS_FILE")
if results_file.exists():
    content = results_file.read_text()
    
    ref_match = re.search(r'beer_reference: ([\d.]+) ms', content)
    cmsis_match = re.search(r'beer_cmsis_nn: ([\d.]+) ms', content) 
    
    if ref_match and cmsis_match:
        ref_time = float(ref_match.group(1))
        cmsis_time = float(cmsis_match.group(1))
        
        improvement = ref_time - cmsis_time
        improvement_pct = (improvement / ref_time) * 100
        
        print(f"Performance Analysis:")
        print(f"  Reference (no CMSIS-NN): {ref_time:.2f} ms")
        print(f"  CMSIS-NN optimized:     {cmsis_time:.2f} ms") 
        print(f"  Improvement:             {improvement:.2f} ms ({improvement_pct:.1f}%)")
        
        if improvement > 0:
            print(f"  🚀 CMSIS-NN optimization is {improvement_pct:.1f}% FASTER!")
        else:
            print(f"  ⚠️  CMSIS-NN optimization is {abs(improvement_pct):.1f}% slower")
    else:
        print("Could not parse timing results")
EOF
else
    echo "❌ No results file found"
fi

echo
echo "=== Comparison Complete ==="
echo "Build artifacts saved in: $BUILD_DIR"
