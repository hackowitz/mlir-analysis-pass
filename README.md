# Known Bits Analysis

An MLIR dataflow analysis as a loadable `mlir-opt` plugin, with no LLVM source tree required and nothing to patch upstream.

The included analysis, `known-bits-analysis`, decides fore each bit which integer values in the LLVM dialect are known to be $\hat{0}$, $\hat{1}$, $\bot$ (neither 1 nor 0, as in unreached code), or $\top$ (either 1 or 0, as in unknown reached values).

## Building

```sh
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

That is the whole procedure on Linux, macOS, and WSL2. There is no platform
flag to set and no path to edit. `CMakeLists.txt` finds MLIR by asking
whichever `llvm-config` is on your `PATH` where its CMake package lives, so if
`mlir-opt` runs, the build should configure.

To build against a specific MLIR instead:

```sh
cmake -S . -B build -DMLIR_DIR=/path/to/prefix/lib/cmake/mlir
```

You need an LLVM built with MLIR enabled and plugins enabled
(`-DLLVM_ENABLE_PROJECTS=mlir -DLLVM_ENABLE_PLUGINS=ON`; both are ordinary on
Linux and macOS). Distribution packages work: on Debian and Ubuntu that is
`libmlir-dev` alongside `llvm-dev`. On macOS, Homebrew's `llvm` is the easy
route if it ships `mlir-opt` for your version; otherwise build LLVM yourself.
The configure step diagnoses the cases it can detect — no MLIR
found, plugins disabled in the host LLVM, or an `mlir-opt` on `PATH` whose
version does not match what you are building against.

## Running

```sh
./run.sh input.mlir
```

`run.sh` locates the plugin whatever it is called on your platform and puts the
annotated listing on stdout. Or invoke `mlir-opt` yourself:

```sh
mlir-opt --load-pass-plugin=build/KnownBitsAnalysis.so \
         --pass-pipeline='builtin.module(known-bits-analysis)' \
         input.mlir -o /dev/null
```

using `build/KnownBitsAnalysis.dylib` on macOS. The pass leaves the IR unchanged and
writes it to stdout as usual; the annotated view goes to stderr, so the two
streams can be redirected independently. Annotations are comments, so the
annotated listing is still valid MLIR. Values at top or bottom are left
unannotated, so that what prints is exactly what was proved.

Get input in the LLVM dialect from C with:

```sh
clang -S -emit-llvm -o - input.c | mlir-translate --import-llvm
```

## What is where

Two files hold the analysis; the rest is reusable scaffolding.

| File                    |                                                             |
| ----------------------- | ----------------------------------------------------------- |
| `KnownBitsDomain.h`     | The abstract domain: the lattice elements and their join.   |
| `KnownBitsAnalysis.cpp` | The transfer function: two rules, plus a default.           |
| `KnownBitsAnalysis.h`   | Ties the domain to MLIR's sparse forward analysis.          |
| `Annotate.{h,cpp}`      | Prints IR with a comment on each value. Domain-agnostic.    |
| `Plugin.cpp`            | The pass, the solver setup, and the `mlir-opt` entry point. |
| `cmake/RunTest.cmake`   | The test runner.                                            |

To build a different analysis, replace `KnownBitsDomain.h` and the transfer
functions in `KnownBitsAnalysis.cpp`. To rename the whole thing, rename the files,
the `known-bits` namespace, and the three places `KnownBitsAnalysis` and `known-bits-analysis`
appear in `CMakeLists.txt` and `Plugin.cpp`.

## Tests

`test/known-bits.mlir` exercises some subset (since $\bot \sqsubseteq \top$) $ of the transfer rules. `test/known-bits.expected` lists facts that must appear in the output, and — with a leading `!` — facts that must not. The negative checks are the ones that matter: an unsound transfer function still produces plausible-looking output, and only a test that pins down what the analysis must *not* claim will catch it.

Note that MLIR's printer renumbers SSA values, so the checks are written
against operation text rather than the names in `known-bits.mlir`. After adding or
reordering operations, regenerate with `./run.sh test/known-bits.mlir`.

## Notes on portability

Most of the platform-specific knowledge lives in `CMakeLists.txt`, next to the
code it affects. The parts worth knowing about:

**The plugin's file name differs.** It is `KnownBitsAnalysis.dylib` on macOS and
`KnownBitsAnalysis.so` on Linux and WSL2. Nothing in this project spells that out:
CMake is asked via `$<TARGET_FILE:KnownBitsAnalysis>`, and `run.sh` probes for both.

**Linking a plugin on macOS needs special flags.** The plugin deliberately
leaves its MLIR symbols undefined, to be resolved from the `mlir-opt` process
that loads it. On macOS that requires `-undefined dynamic_lookup`, which
`include(HandleLLVMOptions)` supplies. The same include also matches LLVM's
RTTI and exception settings, which differ between distribution packages and
local builds and cause link errors or silent ODR violations when they are
wrong. That is also why `project()` enables C: `HandleLLVMOptions` probes flags
with the C compiler and fails if none is configured.

**A plugin only loads into the LLVM it was built against.** The version is
recorded at compile time and checked at load time, so a mismatch is a clear
error rather than a crash. The configure step warns about it earlier still, by
comparing against the `mlir-opt` it finds.

**The test suite needs no shell.** `cmake/RunTest.cmake` is a CMake script
rather than a shell script, so `ctest` depends on nothing the build did not
already require.

**Under WSL2, build on the Linux filesystem.** A tree under `/mnt/c` is
slow enough to be noticeable and does not reliably carry execute bits.
`.gitattributes` forces LF endings, which keeps `run.sh` working when a
repository is cloned by a Windows git and built inside WSL2.

## How the analysis works

`Plugin.cpp` loads three analyses into one solver. `DeadCodeAnalysis` supplies
reachability — without it the solver must assume every branch is taken — and
`SparseConstantPropagation` resolves branch conditions on its behalf. These are
prerequisites for a precise result, not optional extras. `KnownBitsAnalysis` then
propagates known bits through operations and block arguments until the solver
reaches a fixed point, which is when the pass queries it.

The transfer function has rules to abstractly evaluate MLIR operations on values in our abstract domain. 
- Constant and zero operations have all bits known as written, and are the entry to our abstract domain from concrete values - either from constants in the source code or as outputs of `SparseConstantPropagation`. Without some rule of this kind there would be no facts to propagate at all.
- The sprawling nested `if`/`else` and `switch` block in `KnownBitsAnalysis::visitOperation` matches a LLVM operation to the corresponding operator or method of our abstract domain `KnownBitsState`.
- Everything else is unknown. This is always sound, just imprecise - `llvm.load`, for example, produces a value that might be knowable under some other analysis, but all we can guarantee is that it's within $\top$.
- The domain's fourth element, bottom, means "not yet proved reachable"; the solver starts everything there and raises it as facts arrive, which is what makes the fixed-point iteration terminate.

The analysis is intraprocedural. It does not refine facts on branch conditions, so a value tested against zero is not known nonzero on the taken edge. This is a natural extension, but not within scope of the homework.

## My Results

The analysis was run on the `sqlite3` v3.53.4 source code. For convenience and reproducability, [test/sqlite3.c](test/sqlite3.c), [test/sqlite3.ll](test/sqlite3.ll), and [test/sqlite3.mlir](test/sqlite3.mlir) are included in this repository. The analysis results are tracked as [test/sqlite3.known-bits.mlir](test/sqlite3.known-bits.mlir).

Known bit results are appended as comments in the MLIR file, using verilog-style binary representation. `?` represents $\top$ (top), and `!` represents $\bot$ (bottom). `!` in the output indicates an error, since no reached value can be neither 0 nor 1.  For example `// known bits: 8'b000001?!` is an 8-bit value with the first five bits known to be 0, then one bit known to be 1, then one unknown bit, and lastly one impossible/invalid/unreachable bit (which is nonsensical and should not occur). Trivial $\top$ and $\bot$ numbers are omitted for clarity.

The easiest way to explore the results is with `grep`:
```sh
# 1: list all operations producing known bits
grep -Pn 'known bits: \d+.b[01?!]+$' ./test/sqlite3.known-bits.mlir
# 2: count home many operations produce known bits (35648/352774 at time of writing, a little over 10% of all ops)
!1 | wc -l
# 3: Exclude trivial or uninteresting known bits, like those from sign extension.
!1 | grep -Pv '(constant|zext|sext|trunc)'
# 4: Search instead for numbers with _all_ bits known:
grep -Pn '// known bits: \d+.b[01]+$' ./test/sqlite3.known-bits.mlir
# 5: or for unknown bits among known bits
grep -Pn '// known bits: \d+.b[01?!]*[01]+[?!]+[01]+[01?!]*$' ./test/sqlite3.known-bits.mlir
# 6: or for known bits among unknown bits
grep -Pn '// known bits: \d+.b[01?!]*[?!]+[01]+[?!]+[01?!]*$' ./test/sqlite3.known-bits.mlir
```
These can be combined easily to show more specific combinations, for example finding non-trivial constants:
```llvm
%186 = llvm.and %185, %1 : i32 // known bits: 32'b00000000000000000000000000000000
%187 = llvm.icmp "eq" %186, %1 : i32 // known bits: 1'b1
```
These numbers look uninteresting at first, until we realize constant propagation didn't catch them - and `%187` is a branch condition that proves dead code that the dead code analysis didn't find!

We can also prove that sign extension is not necessary when the MSB is known 0. There are over 100 `sext` opeeratons that could be `zext`, 35 of which could be joined with a preceding `zext`, as below:
```llvm
%251 = llvm.load %250 <alignment = 2> : !llvm.ptr -> i8
%252 = llvm.zext %251 : i8 to i32 // known bits: 32'b000000000000000000000000????????
%253 = llvm.sext %252 : i32 to i64 // known bits: 64'b00000000000000000000000000000000000000000000000000000000????????
```

If my analysis might save a hectobyte on 10 billion strangers' phones, I'll have saved about a terabyte worldwide... but I'm still pretty happy about it.
