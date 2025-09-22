#!/usr/bin/env python3
"""Cross-compile the STM32 N6 convolution harness and run it under QEMU."""

from __future__ import annotations

import argparse
import os
import select
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
STM32_DIR = REPO_ROOT / "src/Core/Tensor/Accelerators/stm32n6"
HARNESS_DIR = REPO_ROOT / "tests/stm32n6_qemu"
CMSIS_STUB_DIR = REPO_ROOT / "tests/fixtures/cmsis_stub"
LINKER_SCRIPT = HARNESS_DIR / "stm32n6.ld"
BEER_LIB_PATH = REPO_ROOT / "zig-out" / "beer" / "libzant.a"
BEER_GENERATED_DIR = REPO_ROOT / "generated" / "beer"
BARE_METAL_SOURCES = (
    HARNESS_DIR / "runtime.c",
    HARNESS_DIR / "semihost_arm.c",  # Use ARM semihosting
    HARNESS_DIR / "semihost_arm.S",
    HARNESS_DIR / "common" / "support.c",
    HARNESS_DIR / "main.c",  # Full test
    STM32_DIR / "conv_f32.c",
    STM32_DIR / "ethos_stub.c",
)

BEER_SOURCES = (
    HARNESS_DIR / "runtime.c",
    HARNESS_DIR / "semihost_arm.c",
    HARNESS_DIR / "semihost_arm.S",
    HARNESS_DIR / "common" / "support.c",
    HARNESS_DIR / "beer_main.c",
    STM32_DIR / "conv_f32.c",
    STM32_DIR / "ethos_stub.c",
)

HOST_SOURCES = (
    HARNESS_DIR / "semihost.c",
    HARNESS_DIR / "support.c",
    HARNESS_DIR / "main.c",
    STM32_DIR / "conv_f32.c",
    STM32_DIR / "ethos_stub.c",
)


@dataclass
class BuildCase:
    name: str
    macros: Sequence[str]
    extra_sources: Sequence[Path]
    include_dirs: Sequence[Path]


class ToolchainError(RuntimeError):
    pass


class Toolchain:
    def build(
        self,
        *,
        output: Path,
        base_sources: Sequence[Path],
        macros: Sequence[str],
        extra_sources: Sequence[Path],
        include_dirs: Sequence[Path],
    ) -> None:
        raise NotImplementedError

    def describe(self) -> str:
        raise NotImplementedError

    def default_macros(self) -> Sequence[str]:
        return ()

    def extra_include_dirs(self) -> Sequence[Path]:
        return ()


class ZigToolchain(Toolchain):
    def __init__(self, exe: Path):
        self.exe = exe

    def build(
        self,
        *,
        output: Path,
        base_sources: Sequence[Path],
        macros: Sequence[str],
        extra_sources: Sequence[Path],
        include_dirs: Sequence[Path],
    ) -> None:
        cmd = [
            str(self.exe),
            "cc",
            "-Dtarget=thumb-freestanding",
            "-Dcpu=cortex_m55",
            "-ffreestanding",
            "-fno-builtin",
            "-fno-exceptions",
            "-fno-stack-protector",
            "-Wl,-T," + str(LINKER_SCRIPT),
            "-O2",
            "-g",
            "-nostdlib",
            "-o",
            str(output),
        ]
        for include in (*self.extra_include_dirs(), STM32_DIR, HARNESS_DIR, *include_dirs):
            cmd.extend(["-I", str(include)])
        for macro in macros:
            cmd.append(f"-D{macro}")
        for source in (*base_sources, *extra_sources):
            cmd.append(str(source))
        subprocess.run(cmd, check=True)

    def describe(self) -> str:
        return f"zig cc ({self.exe})"


class ArmGccToolchain(Toolchain):
    def __init__(self, prefix: str):
        self.prefix = prefix
        self.cc = shutil.which(f"{prefix}-gcc")
        if self.cc is None:
            raise ToolchainError(f"unable to locate {prefix}-gcc in PATH")

    def build(
        self,
        *,
        output: Path,
        base_sources: Sequence[Path],
        macros: Sequence[str],
        extra_sources: Sequence[Path],
        include_dirs: Sequence[Path],
    ) -> None:
        cmd = [
            self.cc,
            "-mcpu=cortex-m55",
            "-mthumb",
            "-mfloat-abi=soft",
            "-ffreestanding",
            "-fno-builtin",
            "-fno-exceptions",
            "-fno-stack-protector",
            "-Wall",
            "-Wextra",
            "-Wno-unused-parameter",
            "-O2",
            "-g",
            "-nostdlib",
            "-Wl,-T",
            str(LINKER_SCRIPT),
            "-Wl,--gc-sections",
            "-Wl,--nmagic",
            "-Wl,-Map=" + str(output.with_suffix(".map")),
            "-o",
            str(output),
        ]
        for include in (*self.extra_include_dirs(), STM32_DIR, HARNESS_DIR, *include_dirs):
            cmd.extend(["-I", str(include)])
        for macro in macros:
            cmd.append(f"-D{macro}")
        for source in (*base_sources, *extra_sources):
            cmd.append(str(source))
        cmd.extend(["-Wl,--start-group", "-lgcc", "-Wl,--end-group"])
        output.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(cmd, check=True)

    def describe(self) -> str:
        return f"{self.prefix}-gcc ({self.cc})"


class HostGccToolchain(Toolchain):
    def __init__(self, exe: Path):
        self.exe = exe

    def build(
        self,
        *,
        output: Path,
        base_sources: Sequence[Path],
        macros: Sequence[str],
        extra_sources: Sequence[Path],
        include_dirs: Sequence[Path],
    ) -> None:
        cmd = [
            str(self.exe),
            "-std=c11",
            "-O2",
            "-g",
            "-Wall",
            "-Wextra",
            "-Wno-unused-parameter",
            "-o",
            str(output),
        ]
        for include in (*self.extra_include_dirs(), STM32_DIR, HARNESS_DIR, *include_dirs):
            cmd.extend(["-I", str(include)])
        for macro in macros:
            cmd.append(f"-D{macro}")
        for source in (*base_sources, *extra_sources):
            cmd.append(str(source))
        cmd.append("-lm")
        subprocess.run(cmd, check=True)

    def describe(self) -> str:
        return f"host gcc ({self.exe})"

    def default_macros(self) -> Sequence[str]:
        return ("STM32N6_HOST=1",)


class ClangToolchain(Toolchain):
    def __init__(self, exe: Path):
        self.exe = exe

    def build(
        self,
        *,
        output: Path,
        base_sources: Sequence[Path],
        macros: Sequence[str],
        extra_sources: Sequence[Path],
        include_dirs: Sequence[Path],
    ) -> None:
        cmd = [
            str(self.exe),
            "-Dtarget=thumb-freestanding",
            "-Dcpu=cortex_m55",
            "-mthumb",
            "-mfpu=fpv5-sp-d16",
            "-mfloat-abi=hard",
            "-ffreestanding",
            "-fno-builtin",
            "-fno-exceptions",
            "-fno-stack-protector",
            "-fdata-sections",
            "-ffunction-sections",
            "-O2",
            "-g",
            "-fuse-ld=lld",
            "-nostdlib",
            "-o",
            str(output),
        ]
        for include in (*self.extra_include_dirs(), STM32_DIR, HARNESS_DIR, *include_dirs):
            cmd.extend(["-I", str(include)])
        for macro in macros:
            cmd.append(f"-D{macro}")
        for source in (*base_sources, *extra_sources):
            cmd.append(str(source))
        subprocess.run(cmd, check=True)

    def describe(self) -> str:
        return f"clang ({self.exe})"


def detect_zig(explicit: str | None) -> Path | None:
    candidates: Iterable[str] = ()
    if explicit:
        candidates = (explicit,)
    else:
        zig_env = os.environ.get("ZIG")
        if zig_env:
            candidates = (zig_env,)
        else:
            candidates = ("zig",)
    for candidate in candidates:
        resolved = shutil.which(candidate)
        if resolved:
            return Path(resolved)
    return None


def detect_clang(explicit: str | None) -> Path | None:
    if explicit is None:
        return None
    resolved = shutil.which(explicit)
    if resolved:
        return Path(resolved)
    raise ToolchainError(f"unable to locate clang binary: {explicit}")


def detect_arm_gcc(prefix: str | None) -> str | None:
    prefixes: Iterable[str]
    if prefix:
        prefixes = (prefix,)
    else:
        env_prefix = os.environ.get("ARM_GNU_PREFIX")
        if env_prefix:
            prefixes = (env_prefix,)
        else:
            prefixes = ("arm-none-eabi",)
    for cand in prefixes:
        if shutil.which(f"{cand}-gcc"):
            return cand
    return None


def detect_host_gcc(explicit: str | None) -> Path | None:
    candidates: Iterable[str] = ()
    if explicit:
        candidates = (explicit,)
    else:
        env_candidate = os.environ.get("HOST_GCC")
        if env_candidate:
            candidates = (env_candidate,)
        else:
            candidates = ("gcc", "cc")
    for cand in candidates:
        resolved = shutil.which(cand)
        if resolved:
            return Path(resolved)
    return None


def detect_qemu(explicit: str | None) -> Path | None:
    if explicit:
        resolved = shutil.which(explicit)
        if resolved:
            return Path(resolved)
        return None
    env_value = os.environ.get("QEMU_SYSTEM_ARM")
    if env_value and shutil.which(env_value):
        return Path(shutil.which(env_value))
    resolved = shutil.which("qemu-system-arm")
    if resolved:
        return Path(resolved)
    repo_stub = REPO_ROOT / "scripts" / "qemu-system-arm"
    if repo_stub.exists():
        return repo_stub
    return None


def detect_fvp(explicit: str | None) -> Path | None:
    if explicit:
        resolved = shutil.which(explicit)
        if resolved:
            return Path(resolved)
        return None
    env_value = os.environ.get("FVP_BIN") or os.environ.get("ARM_FVP")
    if env_value and shutil.which(env_value):
        return Path(shutil.which(env_value))
    candidates = (
        "FVP_Corstone_SSE-300_Ethos-U55",
        "FVP_Corstone_SSE-300",
        "FVP_MPS3",
        "FVP_Corstone_SSE-310",
        "FVP_Corstone_SSE-200",
    )
    for cand in candidates:
        resolved = shutil.which(cand)
        if resolved:
            return Path(resolved)
    # Heuristic: common local install paths
    likely_paths: list[Path] = []
    home = Path.home()
    likely_paths.append(home / "FVP_Corstone_SSE-300" / "models" / "Linux64_GCC-9.3" / "FVP_Corstone_SSE-300_Ethos-U55")
    likely_paths.append(home / "FVP_Corstone_SSE-300" / "models" / "Linux64_GCC-9.3" / "FVP_Corstone_SSE-300")
    likely_paths.append(home / "Arm" / "FVP_Corstone_SSE-300" / "models" / "Linux64_GCC-9.3" / "FVP_Corstone_SSE-300_Ethos-U55")
    for p in likely_paths:
        if p.exists() and p.is_file():
            return p
    return None


def find_arm_math_header(explicit: str | None) -> Path:
    candidates: Iterable[Path]
    if explicit:
        candidates = (Path(explicit),)
    else:
        env_candidate = os.environ.get("STM32N6_CMSIS_INCLUDE")
        search_roots: list[Path] = []
        if env_candidate:
            search_roots.append(Path(env_candidate))
        search_roots.extend(
            [
                REPO_ROOT / "third_party" / "CMSIS-DSP" / "Include",
                REPO_ROOT / "third_party" / "CMSIS_5" / "CMSIS" / "DSP" / "Include",
                REPO_ROOT / "third_party" / "CMSIS-NN" / "CMSIS" / "DSP" / "Include",
                CMSIS_STUB_DIR,
            ]
        )
        candidates = search_roots
    for candidate in candidates:
        header = candidate / "arm_math.h"
        if header.exists():
            return candidate
    raise ToolchainError(
        "unable to locate arm_math.h; pass --cmsis-include or set STM32N6_CMSIS_INCLUDE"
    )


def find_arm_nn_header(explicit: str | None, dsp_include: Path) -> Path:
    if explicit:
        candidate = Path(explicit)
        if (candidate / "arm_nnfunctions.h").exists():
            return candidate
        raise ToolchainError(f"arm_nnfunctions.h not found under {candidate}")

    env_candidate = os.environ.get("STM32N6_CMSIS_NN_INCLUDE")
    if env_candidate:
        candidate = Path(env_candidate)
        if (candidate / "arm_nnfunctions.h").exists():
            return candidate

    candidate_roots: list[Path] = []
    if dsp_include != CMSIS_STUB_DIR:
        candidate_roots.append(dsp_include.parent.parent / "NN" / "Include")
        candidate_roots.append(REPO_ROOT / "third_party" / "CMSIS-NN" / "Include")
    candidate_roots.append(CMSIS_STUB_DIR)

    for candidate in candidate_roots:
        if (candidate / "arm_nnfunctions.h").exists():
            return candidate

    raise ToolchainError("unable to locate arm_nnfunctions.h; pass --cmsis-nn-include explicitly")


def find_arm_convolve_source(explicit: str | None, nn_include: Path) -> Path:
    if explicit:
        candidate = Path(explicit)
        if candidate.exists():
            return candidate
        raise ToolchainError(f"arm_convolve_s8 source not found: {candidate}")
    if nn_include == CMSIS_STUB_DIR:
        return CMSIS_STUB_DIR / "arm_convolve_s8.c"

    source_root = nn_include.parent / "Source"
    matches = list(source_root.rglob("arm_convolve_s8.c")) if source_root.exists() else []
    if matches:
        return matches[0]

    stub = CMSIS_STUB_DIR / "arm_convolve_s8.c"
    if stub.exists():
        return stub

    raise ToolchainError("unable to find arm_convolve_s8.c; pass --cmsis-convolve explicitly")


def get_cmsis_sources(convolve_source: Path, nn_include: Path) -> tuple[Path, ...]:
    """Get all required CMSIS-NN source files"""
    sources = [convolve_source]

    # Add additional CMSIS-NN sources if using real CMSIS-NN (not stubs)
    if nn_include != CMSIS_STUB_DIR:
        cmsis_nn_source = nn_include.parent / "Source"
        if cmsis_nn_source.exists():
            additional_sources = [
                # Wrapper and buffer size helpers
                cmsis_nn_source / "ConvolutionFunctions" / "arm_convolve_wrapper_s8.c",
                cmsis_nn_source / "ConvolutionFunctions" / "arm_convolve_get_buffer_sizes_s8.c",
                cmsis_nn_source / "ConvolutionFunctions" / "arm_depthwise_conv_get_buffer_sizes_s8.c",
                # Common conv kernels the wrapper may dispatch to
                cmsis_nn_source / "ConvolutionFunctions" / "arm_convolve_1x1_s8.c",
                cmsis_nn_source / "ConvolutionFunctions" / "arm_convolve_1x1_s8_fast.c",
                cmsis_nn_source / "ConvolutionFunctions" / "arm_convolve_1_x_n_s8.c",
                # Depthwise wrapper and variants
                cmsis_nn_source / "ConvolutionFunctions" / "arm_depthwise_conv_wrapper_s8.c",
                cmsis_nn_source / "ConvolutionFunctions" / "arm_depthwise_conv_s8.c",
                cmsis_nn_source / "ConvolutionFunctions" / "arm_depthwise_conv_s8_opt.c",
                cmsis_nn_source / "ConvolutionFunctions" / "arm_depthwise_conv_3x3_s8.c",
                # MatMul kernels used by conv
                cmsis_nn_source / "ConvolutionFunctions" / "arm_nn_mat_mult_s8.c",
                cmsis_nn_source / "ConvolutionFunctions" / "arm_nn_mat_mult_kernel_s8_s16.c",
                cmsis_nn_source / "ConvolutionFunctions" / "arm_nn_mat_mult_kernel_row_offset_s8_s16.c",
                # Support functions
                cmsis_nn_source / "NNSupportFunctions" / "arm_s8_to_s16_unordered_with_offset.c",
                cmsis_nn_source / "NNSupportFunctions" / "arm_nn_mat_mult_nt_t_s8.c",
                cmsis_nn_source / "NNSupportFunctions" / "arm_nn_vec_mat_mult_t_s8.c",
                cmsis_nn_source / "NNSupportFunctions" / "arm_nn_mat_mult_nt_t_s8_s32.c",
            ]
            # Only add files that exist
            for src in additional_sources:
                if src.exists():
                    sources.append(src)

    return tuple(sources)


def get_cmsis_dsp_sources(dsp_include: Path) -> tuple[Path, ...]:
    if dsp_include == CMSIS_STUB_DIR:
        return ()
    dsp_root = dsp_include.parent
    candidates = [
        dsp_root / "Source" / "BasicMathFunctions" / "arm_dot_prod_f32.c",
    ]
    return tuple(src for src in candidates if src.exists())


def build_cases(dsp_include: Path, nn_include: Path, convolve_source: Path) -> list[BuildCase]:
    include_dirs: list[Path] = []
    include_dirs.append(dsp_include)
    if nn_include != dsp_include:
        include_dirs.append(nn_include)
    
    # Add CMSIS Core include if using real CMSIS headers (not stubs)
    if dsp_include != CMSIS_STUB_DIR:
        cmsis_core = REPO_ROOT / "third_party" / "CMSIS_5" / "CMSIS" / "Core" / "Include"
        if cmsis_core.exists():
            include_dirs.append(cmsis_core)
    
    include_tuple = tuple(include_dirs)
    return [
        BuildCase("reference", (), (), ()),
        BuildCase(
            "helium",
            ("ZANT_HAS_CMSIS_DSP=1", "ZANT_HAS_CMSIS_NN=1"),
            (*get_cmsis_sources(convolve_source, nn_include), *get_cmsis_dsp_sources(dsp_include)),
            include_tuple,
        ),
        BuildCase(
            "ethos",
            ("ZANT_HAS_CMSIS_DSP=1", "ZANT_HAS_CMSIS_NN=1", "ZANT_HAS_ETHOS_U=1"),
            (*get_cmsis_sources(convolve_source, nn_include), *get_cmsis_dsp_sources(dsp_include)),
            include_tuple,
        ),
    ]


def run_qemu(
    qemu: Path,
    elf_path: Path,
    *,
    verbose: bool,
    success_marker: str | None,
    timeout: float,
) -> subprocess.CompletedProcess[str]:
    cmd = [
        str(qemu),
        "-M",
        "mps3-an547",
        "-cpu",
        "cortex-m55",
        "-kernel",
        str(elf_path),
        "-semihosting",
        "-semihosting-config",
        "enable=on,target=auto",
        "-serial",
        "mon:stdio",
        "-nographic",
    ]
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    output_chunks: list[str] = []
    marker_detected = False
    failure_detected = False
    fatal_detected = False
    deadline = time.monotonic() + timeout

    assert process.stdout is not None  # for type checkers

    while True:
        if process.poll() is not None:
            # Process exited on its own; capture remaining output.
            remainder = process.stdout.read()
            if remainder:
                output_chunks.append(remainder)
                if verbose:
                    sys.stdout.write(remainder)
            break

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break

        ready, _, _ = select.select([process.stdout], [], [], max(remaining, 0.0))
        if not ready:
            continue

        line = process.stdout.readline()
        if line == "" and process.poll() is not None:
            break
        if not line:
            continue

        output_chunks.append(line)
        if verbose:
            sys.stdout.write(line)

        if success_marker and success_marker in line:
            marker_detected = True
            break
        if "FAIL" in line:
            failure_detected = True
            break
        if "fatal: Lockup" in line:
            fatal_detected = True
            break

    if marker_detected:
        # Stop QEMU once success marker is seen to avoid waiting for watchdog timeouts.
        process.terminate()
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=1.0)
        return subprocess.CompletedProcess(cmd, 0, "".join(output_chunks), "")

    # Either timed out or saw an explicit failure; ensure QEMU terminates.
    process.terminate()
    try:
        stdout_tail, _ = process.communicate(timeout=2.0)
    except subprocess.TimeoutExpired:
        process.kill()
        stdout_tail, _ = process.communicate()
    if stdout_tail:
        output_chunks.append(stdout_tail)

    stdout_text = "".join(output_chunks)
    exit_code = process.returncode if process.returncode is not None else -1

    if failure_detected or fatal_detected:
        return subprocess.CompletedProcess(cmd, exit_code or 1, stdout_text, "")

    # Timed out or exited without explicit marker; treat as success but annotate return code.
    if exit_code not in (0, None):
        return subprocess.CompletedProcess(cmd, exit_code, stdout_text, "")
    return subprocess.CompletedProcess(cmd, 0, stdout_text, "")


def run_fvp(
    fvp: Path,
    elf_path: Path,
    *,
    verbose: bool,
    success_marker: str | None,
    timeout: float,
) -> subprocess.CompletedProcess[str]:
    # Prefer config file if present
    cfg = REPO_ROOT / "build" / "beer_comparison" / "fvp_config.txt"
    cmd: list[str] = [str(fvp)]
    if cfg.exists():
        cmd.extend(["-f", str(cfg)])
    # Common flags to mirror QEMU behavior
    cmd.extend(
        [
            "-a",
            str(elf_path),
            "-C",
            "cpu0.semihosting-enable=1",
            "-C",
            "mps3_board.uart0.out_file=-",
            "-C",
            "mps3_board.telnetterminal0.start_telnet=0",
            "-C",
            "mps3_board.visualisation.disable-visualisation=1",
        ]
    )

    # Ensure FVP can find its runtime libs (e.g., libpython3.9 from a Conda env)
    env = os.environ.copy()
    ld_paths: list[str] = []
    conda_prefix = env.get("CONDA_PREFIX")
    if conda_prefix:
        ld_paths.append(str(Path(conda_prefix) / "lib"))
    else:
        # Fallback heuristics: common conda locations and env name fvp-py39
        home = Path.home()
        candidates = [
            home / "anaconda3" / "envs" / "fvp-py39" / "lib",
            home / "miniconda3" / "envs" / "fvp-py39" / "lib",
            home / "anaconda3" / "lib",
            home / "miniconda3" / "lib",
            Path("/usr/lib"),
            Path("/usr/lib64"),
        ]
        for c in candidates:
            if c.exists():
                ld_paths.append(str(c))
        # Allow explicit override
        if env.get("FVP_LIB_DIR"):
            ld_paths.insert(0, env["FVP_LIB_DIR"]) 

    # If we still couldn't guess, try to locate a directory that contains libpython3.9
    def dir_contains_libpython39(p: Path) -> bool:
        try:
            for name in ("libpython3.9.so.1.0", "libpython3.9.so"):
                if (p / name).exists():
                    return True
        except Exception:
            pass
        return False
    if not any(dir_contains_libpython39(Path(p)) for p in ld_paths):
        search_roots = [
            Path.home() / "anaconda3" / "envs",
            Path.home() / "miniconda3" / "envs",
            Path.home(),
            Path("/opt"),
        ]
        for root in search_roots:
            try:
                if not root.exists():
                    continue
                # only scan shallow for performance
                for sub in list(root.iterdir())[:50]:
                    libdir = sub / "lib"
                    if libdir.exists() and dir_contains_libpython39(libdir):
                        ld_paths.append(str(libdir))
                        raise StopIteration
            except StopIteration:
                break
    if ld_paths:
        prev = env.get("LD_LIBRARY_PATH", "")
        combined = ":".join(ld_paths + ([prev] if prev else []))
        env["LD_LIBRARY_PATH"] = combined

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )

    output_chunks: list[str] = []
    marker_detected = False
    failure_detected = False
    fatal_detected = False
    deadline = time.monotonic() + timeout

    assert process.stdout is not None
    while True:
        if process.poll() is not None:
            remainder = process.stdout.read()
            if remainder:
                output_chunks.append(remainder)
                if verbose:
                    sys.stdout.write(remainder)
            break

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break

        ready, _, _ = select.select([process.stdout], [], [], max(remaining, 0.0))
        if not ready:
            continue

        line = process.stdout.readline()
        if line == "" and process.poll() is not None:
            break
        if not line:
            continue

        output_chunks.append(line)
        if verbose:
            sys.stdout.write(line)

        if success_marker and success_marker in line:
            marker_detected = True
            break
        if "FAIL" in line:
            failure_detected = True
            break
        if "fatal" in line.lower():
            fatal_detected = True
            break

    if marker_detected:
        process.terminate()
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=1.0)
        return subprocess.CompletedProcess(cmd, 0, "".join(output_chunks), "")

    process.terminate()
    try:
        stdout_tail, _ = process.communicate(timeout=2.0)
    except subprocess.TimeoutExpired:
        process.kill()
        stdout_tail, _ = process.communicate()
    if stdout_tail:
        output_chunks.append(stdout_tail)

    stdout_text = "".join(output_chunks)
    exit_code = process.returncode if process.returncode is not None else -1
    if failure_detected or fatal_detected:
        return subprocess.CompletedProcess(cmd, exit_code or 1, stdout_text, "")
    if exit_code not in (0, None):
        return subprocess.CompletedProcess(cmd, exit_code, stdout_text, "")
    return subprocess.CompletedProcess(cmd, 0, stdout_text, "")


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zig", help="Path to a Zig binary for cross-compilation")
    parser.add_argument("--clang", help="Path to a Clang binary for cross-compilation")
    parser.add_argument(
        "--arm-prefix",
        help="GNU Arm Embedded Toolchain prefix (default: arm-none-eabi)",
    )
    parser.add_argument("--host-gcc", help="Path to a host GCC fallback compiler")
    parser.add_argument("--qemu", help="Path to qemu-system-arm")
    parser.add_argument("--cmsis-include", help="Directory containing arm_math.h")
    parser.add_argument(
        "--cmsis-nn-include",
        help="Directory containing arm_nnfunctions.h (defaults to sibling of --cmsis-include)",
    )
    parser.add_argument(
        "--cmsis-convolve",
        "--cmsis-source",
        dest="cmsis_convolve",
        help="Override the path to arm_convolve_s8.c (defaults to CMSIS or stub)",
    )
    parser.add_argument("--keep-build", action="store_true", help="Keep the build directory")
    parser.add_argument("--verbose", action="store_true", help="Stream QEMU output as it runs")
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Number of times to run each firmware variant when measuring timing",
    )
    parser.add_argument(
        "--run-seconds",
        type=float,
        default=3.0,
        help="Allow each QEMU instance to run for this many seconds before the harness terminates it",
    )
    parser.add_argument(
        "--beer",
        action="store_true",
        help="Build and run the Beer model firmware variants",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)

    if args.repeat <= 0:
        raise ToolchainError("--repeat must be at least 1")

    zig_path = detect_zig(args.zig)
    arm_prefix = detect_arm_gcc(args.arm_prefix)
    clang_path = detect_clang(args.clang)
    host_gcc = detect_host_gcc(args.host_gcc)

    toolchain: Toolchain | None = None
    base_sources: Sequence[Path] | None = None
    if arm_prefix is not None:
        toolchain = ArmGccToolchain(arm_prefix)
        base_sources = BARE_METAL_SOURCES
    elif zig_path is not None:
        toolchain = ZigToolchain(zig_path)
        base_sources = BARE_METAL_SOURCES
    elif clang_path is not None:
        toolchain = ClangToolchain(clang_path)
        base_sources = BARE_METAL_SOURCES
    elif host_gcc is not None:
        toolchain = HostGccToolchain(host_gcc)
        base_sources = HOST_SOURCES
    else:
        raise ToolchainError(
            "no cross compiler available; install Zig 0.14, the GNU Arm Embedded toolchain, or provide --host-gcc"
        )

    assert base_sources is not None

    qemu_path = detect_qemu(args.qemu)
    if qemu_path is None:
        raise ToolchainError(
            "qemu-system-arm not found; install QEMU or set the --qemu flag"
        )

    include_dir = find_arm_math_header(args.cmsis_include)
    nn_include = find_arm_nn_header(args.cmsis_nn_include, include_dir)
    convolve_source = find_arm_convolve_source(args.cmsis_convolve, nn_include)
    cases = build_cases(include_dir, nn_include, convolve_source)

    print(f"Using toolchain: {toolchain.describe()}")
    print(f"Using QEMU: {qemu_path}")
    print(f"CMSIS DSP include path: {include_dir}")
    print(f"CMSIS NN include path: {nn_include}")
    print(f"arm_convolve_s8 source: {convolve_source}")

    build_ctx = (
        tempfile.TemporaryDirectory(prefix="stm32n6-qemu-")
        if not args.keep_build
        else None
    )
    try:
        build_dir = Path(build_ctx.name) if build_ctx is not None else (REPO_ROOT / "build" / "stm32n6_qemu")
        if build_ctx is None:
            build_dir.mkdir(parents=True, exist_ok=True)
        timing_records: list[tuple[BuildCase, list[float]]] = []
        for case in cases:
            elf_path = build_dir / f"stm32n6_{case.name}.elf"
            print(f"\n[build] {case.name}")
            toolchain.build(
                output=elf_path,
                base_sources=base_sources,
                macros=(*toolchain.default_macros(), *case.macros),
                extra_sources=case.extra_sources,
                include_dirs=case.include_dirs,
            )
            durations: list[float] = []
            for iteration in range(args.repeat):
                if args.repeat > 1:
                    print(f"[run]   {case.name} ({iteration + 1}/{args.repeat})")
                else:
                    print(f"[run]   {case.name}")
                start = time.perf_counter()
                expected_marker = f"stm32n6 {case.name} PASS"
                result = run_qemu(
                    qemu_path,
                    elf_path,
                    verbose=args.verbose,
                    success_marker=expected_marker,
                    timeout=args.run_seconds,
                )
                duration = time.perf_counter() - start
                durations.append(duration)
                if args.verbose:
                    sys.stdout.write(result.stdout)
                expected_marker = f"stm32n6 {case.name} PASS"
                if result.returncode not in (0, 1):
                    raise RuntimeError(
                        f"QEMU exited with status {result.returncode} during {case.name} run:\n{result.stdout}"
                    )
                if expected_marker not in result.stdout:
                    raise RuntimeError(
                        f"Harness output missing PASS marker for {case.name}:\n{result.stdout}"
                    )
                if not args.verbose:
                    for line in result.stdout.splitlines():
                        if expected_marker in line:
                            print(line)
                            break
                print(f"✅ {case.name} completed in {duration * 1000.0:.2f} ms")
            timing_records.append((case, durations))
    finally:
        if build_ctx is not None:
            build_ctx.cleanup()

    print("\nAll STM32N6 QEMU cases passed.")
    if timing_records:
        print("\nTiming summary:")
        name_to_avg: dict[str, float] = {}
        for case, durations in timing_records:
            avg = sum(durations) / len(durations)
            best = min(durations)
            name_to_avg[case.name] = avg
            formatted = ", ".join(f"{d * 1000.0:.2f} ms" for d in durations)
            print(
                f"  {case.name}: mean {avg * 1000.0:.2f} ms, min {best * 1000.0:.2f} ms"
                f" over {len(durations)} run(s) [{formatted}]"
            )
        reference_avg = name_to_avg.get("reference")
        helium_avg = name_to_avg.get("helium")
        ethos_avg = name_to_avg.get("ethos")
        if reference_avg is not None:
            for name, avg in name_to_avg.items():
                if name == "reference":
                    continue
                delta = avg - reference_avg
                print(f"    Δ({name} - reference): {delta * 1000.0:.2f} ms")
        if helium_avg is not None and ethos_avg is not None:
            delta = ethos_avg - helium_avg
            print(f"    Δ(ethos - helium): {delta * 1000.0:.2f} ms")

    if args.beer:
        beer_lib = ensure_beer_library()
        beer_cases = [case for case in build_cases(include_dir, nn_include, convolve_source) if case.name in ("reference", "helium")]
        
        # Since the beer library was built with CMSIS-NN calls, both reference and helium cases need CMSIS-NN sources
        cmsis_sources = get_cmsis_sources(convolve_source, nn_include)
        cmsis_dsp_sources = get_cmsis_dsp_sources(include_dir)
        
        # Ensure all beer cases get CMSIS include directories
        beer_include_dirs = [include_dir]
        if nn_include != include_dir:
            beer_include_dirs.append(nn_include)
        # Add CMSIS Core include if using real CMSIS headers (not stubs)
        if include_dir != CMSIS_STUB_DIR:
            cmsis_core = REPO_ROOT / "third_party" / "CMSIS_5" / "CMSIS" / "Core" / "Include"
            if cmsis_core.exists():
                beer_include_dirs.append(cmsis_core)
        
        beer_timing: list[tuple[str, list[float]]] = []
        for case in beer_cases:
            case_name = f"beer_{case.name}"
            elf_path = build_dir / f"{case_name}.elf"
            print(f"\n[build] {case_name}")
            
            # For beer tests, include CMSIS sources only if not already in case.extra_sources
            if case.extra_sources:
                # Helium case already has CMSIS sources, just add beer lib
                extra_sources = (*case.extra_sources, beer_lib)
            else:
                # Reference case needs CMSIS sources added
                extra_sources = (*cmsis_sources, *cmsis_dsp_sources, beer_lib)
            
            # Combine case include dirs with beer-specific CMSIS include dirs
            all_include_dirs = (*beer_include_dirs, *case.include_dirs, BEER_GENERATED_DIR)
            
            toolchain.build(
                output=elf_path,
                base_sources=BEER_SOURCES,
                macros=(*toolchain.default_macros(), *case.macros),
                extra_sources=extra_sources,
                include_dirs=all_include_dirs,
            )

            durations: list[float] = []
            for iteration in range(args.repeat):
                if args.repeat > 1:
                    print(f"[run]   {case_name} ({iteration + 1}/{args.repeat})")
                else:
                    print(f"[run]   {case_name}")
                start = time.perf_counter()
                result = run_qemu(
                    qemu_path,
                    elf_path,
                    verbose=args.verbose,
                    success_marker="beer PASS",
                    timeout=args.run_seconds,
                )
                duration = time.perf_counter() - start
                durations.append(duration)
                if args.verbose:
                    sys.stdout.write(result.stdout)
                if result.returncode not in (0, 1):
                    raise RuntimeError(
                        f"QEMU exited with status {result.returncode} during {case_name} run:\n{result.stdout}"
                    )
                if "beer PASS" not in result.stdout:
                    raise RuntimeError(
                        f"Harness output missing PASS marker for {case_name}:\n{result.stdout}"
                    )
                if not args.verbose:
                    for line in result.stdout.splitlines():
                        if "beer PASS" in line:
                            print(line)
                            break
                print(f"✅ {case_name} completed in {duration * 1000.0:.2f} ms")
            beer_timing.append((case_name, durations))

        print("\nBeer model timing summary:")
        stats: dict[str, float] = {}
        for case_name, durations in beer_timing:
            avg = sum(durations) / len(durations)
            best = min(durations)
            stats[case_name] = avg
            formatted = ", ".join(f"{d * 1000.0:.2f} ms" for d in durations)
            print(
                f"  {case_name}: mean {avg * 1000.0:.2f} ms, min {best * 1000.0:.2f} ms"
                f" over {len(durations)} run(s) [{formatted}]"
            )
        ref_avg = stats.get("beer_reference")
        helium_avg = stats.get("beer_helium")
        if ref_avg is not None and helium_avg is not None:
            delta = helium_avg - ref_avg
            print(f"    Δ(beer_helium - beer_reference): {delta * 1000.0:.2f} ms")
    return 0


def ensure_beer_library() -> Path:
    # Skip zig build - assume library already exists or use ARM GCC build
    # env = os.environ.copy()
    # env.setdefault("ZANT_FBA_SIZE_KB", "320")
    # env.setdefault("ZANT_FBA_SECTION", ".tensor_pool")
    # cmd = [
    #     "zig",
    #     "build",
    #     "lib",
    #     "-Dmodel=beer",
    #     "-Ddynamic=true",
    #     "-Ddo_export=true",
    #     "-Dfuse=true",
    #     "-Dtarget=thumb-freestanding",
    #     "-Dcpu=cortex_m55",
    #     "-Doptimize=ReleaseSmall",
    #     "-Dstm32n6_accel=true",
    #     "-Dstm32n6_use_cmsis=true",
    # ]
    # subprocess.run(cmd, check=True, cwd=REPO_ROOT, env=env)
    
    # Check if beer library exists, if not create a dummy path
    if BEER_LIB_PATH.exists():
        return BEER_LIB_PATH
    else:
        # Return the generated directory path instead - we'll use source files directly
        return BEER_GENERATED_DIR / "lib_beer.zig"


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except ToolchainError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
