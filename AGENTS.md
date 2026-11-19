# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Julia package for simulating atmospheric turbulence effects on imaging systems. Generates
Kolmogorov-statistics phase screens (optionally via Harding interpolation) and propagates them
through a configurable optical pipeline to produce simulated images. Supports CPU multi-threading
and GPU backends (tested with CUDA.jl) via `Adapt.jl`.

## Commands

Run the full test suite:
```
julia --project=test test/runtests.jl
```

Run a single test file (e.g. `test_atmosphere.jl`) without waiting for the full suite:
```
julia --project=test -e 'using Test, AtmosphericTurbulenceSimulator, LinearAlgebra, Statistics, HDF5, ProgressMeter; include("test/test_atmosphere.jl")'
```

When running tests, show all output from `Precompiling packages finished.` onward — everything before it is package-version noise.

## Layout

```
src/
  AtmosphericTurbulenceSimulator.jl  # Module, includes/exports, `speckle_viewer` stub
  utils.jl        # get_num_threads / set_num_threads, device_eigen, diameter, Maybe
  io.jl           # BufferedDataset, read_batch!, write_batch!, HDF5File, open_file, prepare_dataset
  atmosphere.jl   # Phase generation: kolmogorov_covmat, KarhunenLoeveSampler, HardingInterpolator,
                  #   SavedPhaseReader, PhaseBuffers; specs SingleLayer / SavedPhases
  imaging.jl      # Optical pipeline: FilterSpec, ReadoutSpec, Exposure, true-sky models, ImagingSpec,
                  #   CircularAperture, BilinearInterpolator, ImagingPlan, OpticalBuffers,
                  #   forward_pass! / backward_pass!, phases_to_pupil!, pupil_to_psf!, compute_psf!,
                  #   apply_truesky!, readout!
  simulation.jl   # SimulationBuffers, simulation_run!!, simulation_run, public API:
                  #   simulate_phases, simulate_images
  precompile.jl   # PrecompileTools workload
ext/MakieExt.jl  # `speckle_viewer` implementation (Makie extension)
test/             # Test suite (runtests.jl + per-module files)
docs/             # Documenter.jl source
contrib/          # Notebooks and benchmarks (not part of the package)
```

**Include order is load-order constrained**: `utils.jl`, `io.jl`, `atmosphere.jl`, `imaging.jl`,
`simulation.jl`, `precompile.jl`. `io.jl` must come before `atmosphere.jl` because `SavedPhaseReader`
wraps a `BufferedDataset`; `imaging.jl` uses `AtmosphereSpec` and helpers
from `atmosphere.jl`; `simulation.jl` uses both.

## Architecture

### Phase generation pipeline

`SingleLayer` → `PhaseBuffers(spec, plate_size[, d]; grid_step, batch, deviceadapter)` →
`out = allocate_phases(pb)` → `samplephases!(out, pb)` fills `out` with a `(nx, ny, batch)` block of phases.

`PhaseBuffers` does **not** own an output array; the caller allocates it. It groups a
`KarhunenLoeveSampler`, an optional `HardingInterpolator` with its low-resolution intermediate array,
and the ``r_0`` scale factor. The sampler and the interpolator are also usable on their own
(`samplephases!`, `harding_interpolate!`).

`SavedPhases` does not go through `PhaseBuffers`: it is replayed by a `SavedPhaseReader`, which has the
same `allocate_phases` / `samplephases!` interface. The internal `_phase_sampler(spec, plate_size,
grid_step, batch, deviceadapter)` picks `PhaseBuffers` or `SavedPhaseReader` for a spec.

Concrete specs: `SingleLayer` (KL decomposition + optional Harding interpolation),
`SavedPhases` (replays from a dataset, cropping to `plate_size`).

### Imaging pipeline

`ImagingSpec` + `AtmosphereSpec` → `ImagingPlan(atm_spec, img_spec)` (immutable, shared by all workers) and
`OpticalBuffers(plan; batch, nthreads)` (scratch: pupil/focal buffers and FFT plans). Then, per batch:

```
compute_psf!(psf, bufs, plan, phases)   # accumulates the PSF over exposure offsets and wavelengths
apply_truesky!(psf, true_sky, bufs)     # convolve with PointSource / DoubleSystem / TrueSkyImage
readout!(images, psf, img_spec.readout_spec)
```

The caller supplies `psf` and `images` (`imgbuf(bufs)` / `imgbuf(T, bufs)` allocate arrays matching the
buffers). `SimulationBuffers` (in `simulation.jl`, internal) bundles the phase sampler (`PhaseBuffers`
or `SavedPhaseReader`), phases block,
`OpticalBuffers`, plan, `psf` and `images` for the simulation loop; `SimulationBuffers(pb)` is the
phases-only variant. The phase plate is padded beyond the aperture to fit the wind shift over the exposure.

### Threading and device adapters

The worker count is a global setting in `utils.jl`: `get_num_threads()` (default 1),
`set_num_threads(n)`, `set_num_threads(f, n)` (scoped), and `set_num_threads(nothing)` (half of
`Threads.nthreads()`). It is read when `OpticalBuffers` and `HardingInterpolator` are constructed, and each
splits its batch into per-worker chunks. `OpticalBuffers` handles the single-chunk case, and buffers
that are already a per-chunk view, by calling the function directly instead of spawning threads
(`maybethreads`).

`ComputeBackend` and `MultiThreaded` no longer exist. `deviceadapter` is a plain `Adapt` adapter (e.g. `CuArray`;
default `identity` everywhere). `device_eigen(adapter)` detects
whether the KL eigendecomposition can run on the device; otherwise it runs on the host.

### Grid step

`ImagingSpec` stores `grid_step` (physical size of one aperture pixel, in the same units as ``r₀``)
directly. The constructors take `d` (aperture diameter) as a **mandatory positional** argument
(it may be `nothing` when the `grid_step` keyword is given) and derive `grid_step = d / diameter(aperture)`,
where `diameter` requires a square grid; a `grid_step` keyword overrides that value. `simulate_phases`,
`PhaseBuffers` and `SavedPhases` accept `d` as an **optional positional** argument. With neither `d` nor
`grid_step`, `SingleLayer` throws an `ArgumentError`; `SavedPhases` keeps `grid_step = nothing` and takes
the grid step of the imaging setup (a mismatching explicit grid step is an `ArgumentError`). There is no
`ap_step` helper — use
`img_spec.grid_step` directly.

The photon budget and readout are described by `ReadoutSpec` (`ImagingSpec.readout_spec`).

### Simulation loop (`simulation_run!!`)

Drives the loop in `simulation.jl`: `samplephases!(sb.phases, sb.phase_buffers)`, then (if imaging is
enabled) `compute_psf!`, `apply_truesky!`, `readout!`, then `write_batch!` for both `img_bd` and `phs_bd`
(in that order). Both are `BufferedDataset`; passing `BufferedDataset(nothing)` is the no-op when saving
is disabled. `simulation_run` opens the output file, prepares the datasets and calls `simulation_run!!`.

### IO abstraction (`BufferedDataset`)

`BufferedDataset{Dt, Bt}` wraps a dataset and a CPU-side buffer:
- **Write**: `write_batch!(bd, j, data)` — for HDF5 uses `do_write_chunk` (zero-copy); for arrays
  handles boundary truncation.
- **Read**: `read_batch!(dest, bd, j[, ix, iy])` — for HDF5 uses `copyto!(dest, dataset, ix, iy, range)`
  (HDF5.jl overload, no allocation for full batches); partial last batches copy only the available
  frames into `dest` (no NaN-padding — the caller owns responsibility for the tail).

`SavedPhaseReader` wraps a `BufferedDataset` and calls `read_batch!` with stored `crop_indices`
to support replaying a spatially larger saved dataset at a smaller `plate_size`. It is stateful: each
`samplephases!` call advances to the next batch.

## Coding conventions

### Batched operations over loops

Prefer batched array operations instead of scalar loops. If a batched form is impossible,
notify the user before falling back to a loop.

Preferred:
```julia
A .= B[:, range(k+1, length=l)]
```
Not:
```julia
for i in 1:l
    A[:, i] = B[:, i + k]
end
```

### In-place operations in loops

When a loop cannot be avoided, use in-place (`!`) variants and `@views` to avoid allocations:

```julia
@views @. buf[ix, iy, :] += factor * src[ix, iy, :]
```

Use `copy!`, `mul!`, `randn!`, `fill!`, `copyto!` etc. instead of their allocating counterparts.

### GPU compatibility

All performance-critical buffers must be allocated via `similar(existing_array, ...)` or
`Adapt.adapt_storage(deviceadapter, ...)`. Never hard-code `Array{...}(undef, ...)` inside a hot path.

### Function naming

Mutating functions end with `!`. Buffer-preparation functions are named `prepare_*`.
Sampling entry points are named `samplephases!` / `compute_psf!`.

### Alignment

Do not align assignments or other code constructs with extra spaces unless explicitly asked:

```julia
# Not this:
var  = 1
var2 = 2

# This:
var = 1
var2 = 2
```

### Type parameters

Numeric precision flows from the top-level spec structs (e.g. `ImagingSpec{T}`, `SingleLayer{T}`).
Buffers inherit their element type from the spec; do not hard-code `Float64` inside internal functions.

## Tests

**Do not edit test files or add new tests unless explicitly asked.**

The suite includes JET static analysis (`test_jet.jl`); new public functions should be exercisable
without introducing new JET errors. The JET threshold is currently `<= 17` reports (all from
third-party packages); `@test_broken` marks the ideal target of 0.

## New features — plan before implementing

When asked to develop a new feature, **do not change any code**. Instead, write out a
step-by-step implementation plan and stop. Include:

- Which files and functions will be added or modified.
- The proposed public API (types, function signatures, keyword arguments).
- Any non-obvious design decisions or trade-offs.
- Potential performance concerns and how they will be addressed.

Only proceed with implementation after the plan has been explicitly approved.

## HDF5 output

Datasets are written in chunked form (chunk = one batch) via `BufferedDataset`. When adding
new output fields, follow the `prepare_dataset` / `write_batch!` pattern in `simulation_run` to
keep I/O zero-copy. For reading back saved data, use `read_batch!` with appropriate crop indices.