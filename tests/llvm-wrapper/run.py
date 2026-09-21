#!/usr/bin/env python3
"""Compile both LLVM 21 wrappers and check attribute semantics without CUDA."""
import os
from pathlib import Path
import shlex
import subprocess
import tempfile

root = Path(__file__).resolve().parents[2]
wrapper = root / 'crates/rustc_codegen_nvvm/rustc_llvm_wrapper'
config = os.environ.get('LLVM_CONFIG_21', 'llvm-config-21')


def llvm_flags(*args):
    return shlex.split(subprocess.check_output([config, *args], text=True))


version = subprocess.check_output([config, '--version'], text=True).strip()
if not version.startswith('21.'):
    raise RuntimeError(f'expected LLVM 21, got {version}')
cxx = shlex.split(os.environ.get('CXX', 'c++'))
flags = [*llvm_flags('--cxxflags'), '-DLLVM_VERSION_MAJOR=21',
         '-DLLVM_COMPONENT_NVPTX', '-I' + str(wrapper)]
with tempfile.TemporaryDirectory(prefix='llvm-wrapper-test-') as directory:
    out = Path(directory)
    for source in ('RustWrapper', 'PassWrapper'):
        subprocess.run([*cxx, *flags, '-c', str(wrapper / (source + '.cpp')),
                        '-o', str(out / (source + '.o'))], check=True)
    executable = out / 'attributes'
    subprocess.run([*cxx, *flags, str(Path(__file__).with_name('attributes.cpp')),
                    str(out / 'RustWrapper.o'),
                    *llvm_flags('--ldflags', '--libs', '--system-libs'),
                    '-o', str(executable)], check=True)
    subprocess.run([str(executable)], check=True)
print('LLVM wrapper compilation and capture attribute checks passed')
