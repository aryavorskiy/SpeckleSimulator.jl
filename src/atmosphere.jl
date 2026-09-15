using LinearAlgebra, HDF5, Random, ChunkSplitters
import Adapt: adapt_storage

const DEFAULT_WAVELEN = 550nm
abstract type AtmosphereSpec{T} end

_as_wavelength(x::Unitful.Length) = x
_as_wavelength(x::Unitful.Quantity) =
    throw(ArgumentError("Wavelengths must be length quantities, got $(typeof(x))."))
_as_wavelength(x::Number) = x * nm
numtype(x) = typeof(ustrip(x))
convert_numtype(::Type{T}, x) where T = convert.(T, ustrip.(x)) .* unit(eltype(x))

"""
    kolmogorov_covmat(W[, r₀_px])
    kolmogorov_covmat([T, ]size[, r₀_px])

Compute the phase covariance matrix of a turbulent layer in the atmosphere, following the Kolmogorov model. The piston
term is excluded in this model.

# Arguments
- `T`: number type for the covariance matrix (default matches `r₀` if provided, otherwise `eltype(W)` or `Float64`).
- `W`: the aperture function as a 2D array of weights. Normalized to `sum(W) == 1` internally.
- `size`: a tuple `(nx, ny)` specifying the size of the aperture function.
- `r₀_px`: Fried parameter (``r_0``) in pixel units (default `1.0`).
"""
function kolmogorov_covmat(W::AbstractMatrix, r₀_px::Number=one(eltype(W)))
    I = eachindex(IndexCartesian(), W)
    C = similar(W, float(typeof(r₀_px)), length(I), length(I))
    for i in 1:length(I), j in 1:length(I)
        x = I[i][1] - I[j][1]
        y = I[i][2] - I[j][2]
        C[i, j] = -0.5 * 6.88 * ((x^2 + y^2) / r₀_px^2)^(5/6)
    end
    Wp = oftype.(float(r₀_px), W ./ sum(W))
    Cp = vec(sum(C .* vec(Wp)', dims=2))
    Cc = sum(Cp .* vec(Wp))
    return C .- (Cp .+ Cp') .+ Cc
end
kolmogorov_covmat(::Type{T}, sz::NTuple{2,Int}, r₀::Number=1.0) where T = kolmogorov_covmat(ones(sz), convert(T, r₀))
kolmogorov_covmat(sz::NTuple{2,Int}, r₀::Number=1.0) = kolmogorov_covmat(typeof(r₀), sz, r₀)

"""
    scale_phases!(phases[, d], r₀)
    scale_phases!(phases, r₀[; grid_step])

Rescale `phases` in place to Fried parameter `r₀`, assuming they were generated with ``r_0 = 1`` pixel.

# Arguments
- `phases`: a `(nx, ny, nframes)` array of phase screens, modified in place.
- `d`: aperture diameter in the same units as `r₀`. If not a square array, the largest dim
  is used to compute the pixel scale.
- `r₀`: target Fried parameter.

# Keyword Arguments
- `grid_step`: physical size of one pixel; defaults to one (i. e. ``r_0`` is in pixels).
"""
function scale_phases!(phases::AbstractArray, d, r₀::Number)
    grid_step = d / maximum(size(phases)[1:2])
    phases .*= convert(eltype(phases), (NoUnits(r₀ / grid_step))^(-5/6))
end
scale_phases!(phases::AbstractArray, r₀::Number; grid_step=1) =
    scale_phases!(phases, maximum(size(phases)[1:2]) * grid_step, r₀)

const EigenType = Union{Tuple{<:Any,<:Any}, Eigen}

"""
    KarhunenLoeveSampler(shape, covmat, batch)
    KarhunenLoeveSampler(shape, (E, U), batch)

Draw phase screens by a Karhunen-Loeve expansion of a phase covariance matrix.

Holds precomputed constants and scratch space only — the caller owns the output array. Use
[`samplephases!`](@ref) to draw a batch into an array of your own.

# Arguments
- `covmat`: an `N×N` phase covariance matrix with `N == prod(shape)`, e.g. from
  [`kolmogorov_covmat`](@ref). Any symmetric covariance works, not only Kolmogorov.
- `(E, U)`: a precomputed eigendecomposition of `covmat`. **`E` is modified in place** —
  negative eigenvalues are clamped to zero.
- `shape`: the `(nx, ny)` shape of one phase screen.
- `batch`: number of screens drawn per [`samplephases!`](@ref) call.
"""
struct KarhunenLoeveSampler{MT}
    shape::NTuple{2,Int}
    noise_transform::MT
    noise_buffer::MT
end
function KarhunenLoeveSampler(sz::NTuple{2,Int}, (E, U)::EigenType, batch::Int)
    @assert length(E) == prod(sz)
    @assert size(U) == (length(E), length(E))
    E .= max.(E, zero(eltype(E)))
    noise_transform = U .* sqrt.(E')
    noise_buffer = similar(U, size(U, 2), batch)
    KarhunenLoeveSampler(sz, noise_transform, noise_buffer)
end
KarhunenLoeveSampler(sz::NTuple{2,Int}, covmat::AbstractMatrix, batch::Int) =
    KarhunenLoeveSampler(sz, eigen(Symmetric(covmat)), batch)
plate_size(sampler::KarhunenLoeveSampler) = sampler.shape
batch_length(sampler::KarhunenLoeveSampler) = size(sampler.noise_buffer, 2)
allocate_phases(sampler::KarhunenLoeveSampler) =
    similar(sampler.noise_buffer, (plate_size(sampler)..., batch_length(sampler)))

"""
    samplephases!(out, sampler::KarhunenLoeveSampler[, α, β])

Draw a batch of phase screens from the given Karhunen-Loeve decomposition.

# Arguments
- `out`: an `(nx, ny, batch)` array to hold the output phase screens. Its size must match
  `sampler.shape` and its third dimension must match `sampler.noise_buffer`.
- `sampler`: a `KarhunenLoeveSampler` instance.
- `α` and `β`: optional scaling factors for the output. If provided, the result is computed as
  `out = α * new_phases + β * out`.
"""
function samplephases!(out::AbstractArray{<:Any,3}, sampler::KarhunenLoeveSampler, α=true, β=false)
    size(out)[1:2] != sampler.shape &&
        throw(DimensionMismatch(lazy"Output array has size $(size(out)[1:2]), but sampler expects $(sampler.shape)"))
    randn!(sampler.noise_buffer)
    mul!(reshape(out, :, size(out, 3)), sampler.noise_transform, sampler.noise_buffer, α, β)
    return out
end

"""
    SingleLayer([T, ]r0[; base_wavelength, wind_velocity, harding])

An `AtmosphereSpec` that produces independent (uncorrelated) phase frames for each timestep.

# Arguments
- `T`: the number type for phase screens (default `Float64`).
- `r0`: Fried parameter (``r_0``) in length units (e.g. `0.2m`). A plain number is interpreted
    in whatever units `d`/`grid_step` uses.

# Keyword Arguments
- `base_wavelength`: reference wavelength used to scale phase screens when a multi-wavelength
  `FilterSpec` is used in length units (default 550 nm). A plain number is assumed to be
  nanometers.
- `wind_velocity`: two-component `(vx, vy)` wind velocity used for long-exposure offsets in
  imaging simulations in velocity units (default `(0, 0)`). A plain number is assumed to be
  in whatever units `d`/`grid_step` and `exposure` use.
- `harding`: keyword arguments passed to [`HardingInterpolator`](@ref) for optional multi-pass
  interpolation of low-resolution phase screens. Defaults to `nsteps=:auto` and `size_heuristics=1024`.
"""
struct SingleLayer{T<:Number,T2<:Number,T3<:Number,KT} <: AtmosphereSpec{T}
    r₀::T
    base_wavelength::T2
    wind_velocity::NTuple{2,T3}
    harding_kw::KT
end
function SingleLayer(r0::Number; base_wavelength=DEFAULT_WAVELEN, wind_velocity=(0, 0), harding=(;))
    wl = _as_wavelength(base_wavelength)
    SingleLayer(float(r0), convert_numtype(numtype(float(r0)), wl), wind_velocity, harding)
end
SingleLayer(::Type{T}, r0::Number; kw...) where T = SingleLayer(convert_numtype(T, r0); kw...)
function prepare_phasebuffers(spec::SingleLayer, plate_size::NTuple{2,Int}, grid_step::Number,
        batch::Int, deviceadapter)
    T = numtype(spec.r₀)
    interpolator = HardingInterpolator(T, plate_size, batch, deviceadapter; spec.harding_kw...)
    low_size = first(interpolator.stage_sizes)
    covar_host = kolmogorov_covmat(T, low_size)
    if device_eigen(deviceadapter)
        covar = adapt_storage(deviceadapter, covar_host)
        E, U = eigen(Symmetric(covar))
    else
        E_host, U_host = eigen(Symmetric(covar_host))
        E = adapt_storage(deviceadapter, E_host)
        U = adapt_storage(deviceadapter, U_host)
    end
    kl = KarhunenLoeveSampler(low_size, (E, U), batch)
    r₀_px = convert(T, NoUnits(spec.r₀ / grid_step))
    nstages(interpolator) == 0 && return PhaseBuffers(kl, nothing, nothing, r₀_px^(-5//6))
    low_array = adapt_storage(deviceadapter, zeros(T, low_size..., batch))
    return PhaseBuffers(kl, interpolator, low_array, r₀_px^(-5//6))
end

# One rung of the ladder: run every pass, then crop the padding and copy out, scaled by `α`.
function _harding_ladder!(to, work_bufs, r₀_px, from, α=true, β=false)
    N = length(work_bufs)
    harding_upsample!(work_bufs[1], from, r₀_px)
    for i in 2:N
        prev_buffer = work_bufs[i - 1]
        harding_upsample!(work_bufs[i], @view(prev_buffer[6:end-5, 6:end-5, :]), r₀_px)
    end
    out_buffer = work_bufs[N]
    crop_offset = (size(out_buffer)[1:2] .- size(to)[1:2]) .÷ 2
    src = view(out_buffer,
        crop_offset[1] + 1:crop_offset[1] + size(to, 1),
        crop_offset[2] + 1:crop_offset[2] + size(to, 2), :)
    isone(α) && iszero(β) ? copyto!(to, src) : (@. to = α * src + β * to)
    return to
end

"""
    harding_upsample!(to, from[, r₀_px=1; rescale=true])

One pass of Harding interpolation: the `(n, m, batch)` phase screen `from` is upsampled onto the
`(2n-1, 2m-1, batch)` grid `to`.

`r₀_px` is the Fried parameter of `from` **in pixels** (default `1`). With `rescale=true` (the default)
the output is scaled by ``2^{5/6}`` so that `to` has the *same* `r₀_px` in pixels as `from` — which is
what makes every pass of a multi-pass ladder identical. With `rescale=false` the physical screen is
preserved instead and `r₀_px` doubles, so a ladder must then decay its injected noise by ``2^{5/6}``
per pass. The two conventions differ by exactly ``2^{5N/6}`` after `N` passes.

The outermost 5 rows and columns of `to` are left uninterpolated — their stencils would run off the
grid — which is why [`harding_interpolate!`](@ref) crops `[6:end-5, 6:end-5, :]` between passes.

Reference: C. M. Harding, R. A. Johnston and R. G. Lane, "Fast simulation of a Kolmogorov phase
screen", Applied Optics 38(11), 1999.
"""
function harding_upsample!(to, from, r₀_px::Number=1; rescale::Bool=true)
    size(to)[1:2] == 2 .* size(from)[1:2] .- 1 || throw(DimensionMismatch(
        "Harding upsampling maps a $(size(from)[1:2]) grid to $(2 .* size(from)[1:2] .- 1), " *
        "got a destination of size $(size(to)[1:2])"))
    size(to, 3) == size(from, 3) || throw(DimensionMismatch(
        "Batch length mismatch: destination has $(size(to, 3)), source $(size(from, 3))"))
    T = eltype(to)
    c_d = convert(T, 0.3198)
    c_m = convert(T, -0.0341)
    c_f = convert(T, -0.0017)
    scale = rescale ? convert(T, exp2(5/6)) : one(T)
    noise_std = convert(T, scale * sqrt(0.5265 / r₀_px^(5/3)))

    # Padding offset
    n, m = size(from)
    inds_odd_x = range(5, length=n-4, step=2)
    inds_odd_y = range(5, length=m-4, step=2)
    inds_even_x = range(4, length=n-3, step=2)
    inds_even_y = range(4, length=m-3, step=2)

    # Copy low-res (scaled, so the rescaling propagates through every stencil below for free)
    @views if rescale
        @. to[1:2:end, 1:2:end, :] = scale * from
    else
        copy!(to[1:2:end, 1:2:end, :], from)
    end

    # Interpolate checker pattern sites
    randn!(@view to[inds_even_x, inds_even_y, :])
    @views @. to[inds_even_x, inds_even_y, :] =
        noise_std * to[inds_even_x, inds_even_y, :] +
        c_d * (to[inds_even_x .+ 1, inds_even_y .+ 1, :] + to[inds_even_x .+ 1, inds_even_y .- 1, :] +
               to[inds_even_x .- 1, inds_even_y .+ 1, :] + to[inds_even_x .- 1, inds_even_y .- 1, :]) +
        c_m * ((to[inds_even_x .+ 3, inds_even_y .+ 1, :] + to[inds_even_x .+ 3, inds_even_y .- 1, :] +
               to[inds_even_x .- 3, inds_even_y .+ 1, :] + to[inds_even_x .- 3, inds_even_y .- 1, :]) +
               (to[inds_even_x .+ 1, inds_even_y .+ 3, :] + to[inds_even_x .+ 1, inds_even_y .- 3, :] +
               to[inds_even_x .- 1, inds_even_y .+ 3, :] + to[inds_even_x .- 1, inds_even_y .- 3, :])) +
        c_f * (to[inds_even_x .+ 3, inds_even_y .+ 3, :] + to[inds_even_x .+ 3, inds_even_y .- 3, :] +
               to[inds_even_x .- 3, inds_even_y .+ 3, :] + to[inds_even_x .- 3, inds_even_y .- 3, :])

    # Fill remaining sites
    noise_std_2 = convert(T, noise_std * exp2(-5/12))
    randn!(@view to[inds_odd_x, inds_even_y, :])
    @views @. to[inds_odd_x, inds_even_y, :] =
        noise_std_2 * to[inds_odd_x, inds_even_y, :] +
        c_d * (to[inds_odd_x, inds_even_y .+ 1, :] + to[inds_odd_x, inds_even_y .- 1, :] +
                to[inds_odd_x .+ 1, inds_even_y, :] + to[inds_odd_x .- 1, inds_even_y, :]) +
        c_m * (to[inds_odd_x .+ 1, inds_even_y .+ 2, :] + to[inds_odd_x .+ 1, inds_even_y .- 2, :] +
                to[inds_odd_x .- 1, inds_even_y .+ 2, :] + to[inds_odd_x .- 1, inds_even_y .- 2, :] +
                to[inds_odd_x .+ 2, inds_even_y .+ 1, :] + to[inds_odd_x .+ 2, inds_even_y .- 1, :] +
                to[inds_odd_x .- 2, inds_even_y .+ 1, :] + to[inds_odd_x .- 2, inds_even_y .- 1, :]) +
        c_f * (to[inds_odd_x .+ 3, inds_even_y, :] + to[inds_odd_x .- 3, inds_even_y, :] +
                to[inds_odd_x, inds_even_y .+ 3, :] + to[inds_odd_x, inds_even_y .- 3, :])

    randn!(@view to[inds_even_x, inds_odd_y, :])
    @views @. to[inds_even_x, inds_odd_y, :] =
        noise_std_2 * to[inds_even_x, inds_odd_y, :] +
        c_d * (to[inds_even_x .+ 1, inds_odd_y, :] + to[inds_even_x .- 1, inds_odd_y, :] +
                to[inds_even_x, inds_odd_y .+ 1, :] + to[inds_even_x, inds_odd_y .- 1, :]) +
        c_m * (to[inds_even_x .+ 1, inds_odd_y .+ 2, :] + to[inds_even_x .+ 1, inds_odd_y .- 2, :] +
                to[inds_even_x .- 1, inds_odd_y .+ 2, :] + to[inds_even_x .- 1, inds_odd_y .- 2, :] +
                to[inds_even_x .+ 2, inds_odd_y .+ 1, :] + to[inds_even_x .+ 2, inds_odd_y .- 1, :] +
                to[inds_even_x .- 2, inds_odd_y .+ 1, :] + to[inds_even_x .- 2, inds_odd_y .- 1, :]) +
        c_f * (to[inds_even_x .+ 3, inds_odd_y, :] + to[inds_even_x .- 3, inds_odd_y, :] +
                to[inds_even_x, inds_odd_y .+ 3, :] + to[inds_even_x, inds_odd_y .- 3, :])
end

"""
    ComputeBackend([adapter; nthreads, device_eigen=true])

Device adapter that enables multi-threaded phase generation and imaging on CPU.

# Arguments
- `adapter`: optional storage adapter for internal buffers.

# Keyword Arguments
- `nthreads`: number of threads to use (default: `Threads.nthreads()` if `adapter` is not
    provided, otherwise 1).
- `device_eigen`: whether the Karhunen-Loeve eigendecomposition is computed on device
    (default `true`).
"""
struct ComputeBackend{AT}
    adapter::AT
    nthreads::Int
    device_eigen::Bool
    ComputeBackend(adapter, nthreads::Int, device_eigen::Bool) =
        new{typeof(adapter)}(adapter, nthreads, device_eigen)
    ComputeBackend(::Type{T}, nthreads::Int, device_eigen::Bool) where {T} =
        new{Val{T}}(Val(T), nthreads, device_eigen)
end
ComputeBackend(adapter=identity;
    nthreads::Int=adapter === identity ? Threads.nthreads() : 1,
    device_eigen::Bool=true) =
    ComputeBackend(adapter, nthreads, device_eigen)
adapt_storage(am::ComputeBackend{AT}, x) where {AT} = adapt_storage(am.adapter, x)
adapt_storage(::ComputeBackend{Val{AT}}, x) where {AT} = adapt_storage(AT, x)
device_eigen(adapter::ComputeBackend) = adapter.device_eigen
device_eigen(::Any) = false

MultiThreaded(adapter, nthreads::Int=1) = ComputeBackend(adapter, nthreads, false)
MultiThreaded(nthreads::Int=Threads.nthreads()) = ComputeBackend(identity, nthreads, false)

function _harding_spec(final_size::NTuple{2,Int}; nsteps=:auto, size_from=nothing, size_heuristics=1024)
    interp_sizes = Tuple{Int,Int}[]
    if size_from !== nothing
        any(size_from .≤ 11) &&
            throw(ArgumentError("`size_from` dimensions must be greater than 11."))
        interpolated_size = size_from
        push!(interp_sizes, interpolated_size)
        n = 0
        while any(final_size .> interpolated_size)
            interpolated_size = 2 .* interpolated_size .- 11
            n += 1
            push!(interp_sizes, interpolated_size)
        end
        return interp_sizes
    elseif nsteps isa Number
        nsteps < 0 && throw(ArgumentError("`nsteps` must be nonnegative."))
        size_from = cld.(final_size .- 11, 2^nsteps) .+ 11
        return _harding_spec(final_size; size_from)
    elseif nsteps === :auto
        size_heuristics = max(size_heuristics, 256)
        size_from = final_size
        while prod(size_from) > size_heuristics
            size_from = cld.(size_from .- 11, 2) .+ 11
        end
        return _harding_spec(final_size; size_from)
    else
        throw(ArgumentError("`nsteps` must be a Number or :auto"))
    end
end
"""
    HardingInterpolator([T, ]size_to, batch[, deviceadapter]; r₀=1[, nsteps, size_from, size_heuristics=1024])

Upsample low-resolution phase screens to `hspec.size_to` with `hspec.nsteps` passes of Harding
interpolation, threaded over the batch dimension.

Holds scratch buffers only — the caller owns both the input and the output; see
[`harding_interpolate!`](@ref).

# Arguments
- `T`: element type of the scratch buffers (default `Float64`).
- `size_to`: the size of the output phase screens.
- `batch`: number of screens interpolated per call.
- `deviceadapter`: storage adapter or [`ComputeBackend`](@ref) for the scratch buffers. Its thread
  count sets how many batch chunks are interpolated in parallel.

# Keyword Arguments
- `nsteps`: number of passes of Harding interpolation. If set to `:auto`, the number of passes
  will be chosen automatically (default).
- `size_from`: alternatively, specify the low-res grid size directly. This must be
  greater than `(11, 11)` in each dimension.
- `size_heuristics`: when `nsteps=:auto`, the maximum allowed number of pixels
  in the low-res grid. Tweak this based on the capability of your hardware to compute `eigen`
  of a `N×N` matrix, where `N` is the number of pixels in the low-res grid.
"""
struct HardingInterpolator{WT,CT}
    size_to::NTuple{2,Int}
    stage_sizes::Vector{NTuple{2,Int}}
    work_bufs::Vector{WT}
    chunk_ranges::Vector{CT}
end
function HardingInterpolator(::Type{T}, size_to::NTuple{2,Int}, batch::Int,
        adapter::ComputeBackend=ComputeBackend(); kw...) where T
    stage_sizes = _harding_spec(size_to; kw...)
    low_size = first(stage_sizes)
    any(low_size .≤ 11) && throw(ArgumentError("Dimensions must be greater than 11"))
    nbufs = min(adapter.nthreads, batch)
    chunk_ranges = collect(chunks(1:batch; n=nbufs))
    work_bufs = map(chunk_ranges) do chunk_range
        map(2:length(stage_sizes)) do j
            adapt_storage(adapter, zeros(T, (stage_sizes[j] .+ 10)..., length(chunk_range)))
        end
    end
    return HardingInterpolator(size_to, stage_sizes, work_bufs, chunk_ranges)
end
HardingInterpolator(::Type{T}, size_to::NTuple{2,Int}, batch::Int, adapter; kw...) where T =
    HardingInterpolator(T, size_to, batch, ComputeBackend(adapter); kw...)
HardingInterpolator(size_to::NTuple{2,Int}, batch::Int, adapter=ComputeBackend(); kw...) =
    HardingInterpolator(Float64, size_to, batch, adapter; kw...)
plate_size(hi::HardingInterpolator) = hi.size_to
batch_length(hi::HardingInterpolator) = sum(length, hi.chunk_ranges)
nstages(hi::HardingInterpolator) = length(hi.stage_sizes) - 1

"""
    harding_interpolate!(out, hi::HardingInterpolator, low[, r₀_px=1, α, β])

Run the full Harding ladder, interpolating low-resolution phase screens into higher ones.

# Arguments
- `out`: a `(nx_high, ny_high, batch)` array to hold the upscaled phasescreens.
- `hi`: a [`HardingInterpolator`](@ref) holding scratch buffers and ladder parameters.
- `low`: a `(nx_low, ny_low, batch)` array of low-resolution phase screens.
- `r₀_px`: Fried parameter of the low-resolution screens in pixels (default `1`). The
  high-resolution output will have the same `r₀_px` in pixels.
- `α` and `β`: optional scaling factors for the output. If provided, the result is computed as
  ``out = α ⋅ harding_interpolate(low) + β ⋅ out``.
"""
function harding_interpolate!(out, hi::HardingInterpolator, low, r₀_px=1, α=true, β=false)
    if length(hi.work_bufs) == 1
        _harding_ladder!(out, only(hi.work_bufs), r₀_px, low, α, β)
    else
        Threads.@threads for i in eachindex(hi.work_bufs)
            chunk = hi.chunk_ranges[i]
            _harding_ladder!(view(out, :, :, chunk), hi.work_bufs[i], r₀_px,
                view(low, :, :, chunk), α, β)
        end
    end
    return out
end

"""
    SavedPhases(dataset[, d]; wind_velocity, base_wavelength, grid_step)

An `AtmosphereSpec` that reuses phase screens saved in a dataset.

The dataset must match the phase-screen layout written by [`simulate_phases`](@ref) and
[`simulate_images`](@ref). Frames are read in order; if more frames are requested than
available, an error is thrown, there is no guarantee what is in the tail of the last batch.

# Arguments
- `dataset`: a 3D array-like containing phase screens with dimensions `(nx, ny, nframes)`.
  This can be an in-memory array or an HDF5 dataset (e.g. `HDF5.Dataset`).
- `d`: aperture diameter of the saved phase screens (in the same units as ``r_0``). Optional;
  when omitted grid step defaults to `1` (one pixel = one unit). Overriden by `grid_step`.

# Keyword Arguments
- `base_wavelength`: reference wavelength in nm used to scale phase screens when a multi-wavelength
  `FilterSpec` is used (default 550 nm).
- `wind_velocity`: two-component `(vx, vy)` wind velocity used for long-exposure offsets in
  imaging simulations (default `(0, 0)`).
- `grid_step`: physical size of one aperture pixel in the same units as ``r_0``. Overrides the
  value derived from `d` when provided.
"""
struct SavedPhases{T<:Real,D,WT,WL,GT} <: AtmosphereSpec{T}
    dataset::D
    wind_velocity::NTuple{2,WT}
    base_wavelength::WL
    grid_step::GT
end
function SavedPhases(dataset, d::Union{Number,Nothing}=nothing;
        wind_velocity::NTuple{2,<:Number}=(0, 0), base_wavelength=DEFAULT_WAVELEN,
        grid_step=nothing)
    T = eltype(dataset)
    WT = typeof(wind_velocity[1])
    base_wl = _as_wavelength(base_wavelength)
    grid_step_final = grid_step !== nothing ? grid_step :
        d !== nothing ? d / maximum(size(dataset)[1:2]) : nothing
    return SavedPhases{T,typeof(dataset),WT,typeof(base_wl),typeof(grid_step_final)}(
        dataset, wind_velocity, base_wl, grid_step_final)
end

"""
    SavedPhaseReader(bd::BufferedDataset, crop_indices, batch[, batch_idx=1])

Replay consecutive batches of `batch` phase screens from a dataset, cropping each frame to
`crop_indices`. Stateful: every [`samplephases!`](@ref) call advances to the next batch.
"""
mutable struct SavedPhaseReader{BDT, CT}
    bd::BDT
    crop_indices::CT
    batch::Int
    batch_idx::Int
end
SavedPhaseReader(bd, crop_indices, batch::Int) = SavedPhaseReader(bd, crop_indices, batch, 1)
plate_size(reader::SavedPhaseReader) = length.(reader.crop_indices)
batch_length(reader::SavedPhaseReader) = reader.batch
allocate_phases(reader::SavedPhaseReader) =
    similar(reader.bd.dataset, (plate_size(reader)..., batch_length(reader)))

"""
    samplephases!(out, reader::SavedPhaseReader[, α])

Read the next batch of saved phase screens into `out` and advance the read cursor. `α` scales the
result; it defaults to `true` (no scaling), since saved screens are already physical.
"""
function samplephases!(out, reader::SavedPhaseReader, α=true)
    read_batch!(out, reader.bd, reader.batch_idx,
        reader.crop_indices[1], reader.crop_indices[2])
    reader.batch_idx += 1
    α === true || (out .*= α)
    return out
end

function prepare_phasebuffers(spec::SavedPhases{T}, plate_size::NTuple{2,Int},
        grid_step::Number, batch::Int, deviceadapter) where T
    spec.grid_step !== nothing && (spec.grid_step != grid_step) && throw(ArgumentError(
        "Saved phase dataset has grid step $(spec.grid_step), but $grid_step was requested."
    ))
    saved_size = size(spec.dataset)::NTuple{3,Int}
    saved_plate_size = saved_size[1:2]
    all(saved_plate_size .>= plate_size) || throw(ArgumentError(
        "Saved phase dataset has plate size $saved_plate_size, but at least $plate_size was requested."
    ))
    crop_indices = (1:plate_size[1], 1:plate_size[2])
    reader = SavedPhaseReader(BufferedDataset(spec.dataset, batch), crop_indices, batch)
    return PhaseBuffers(reader, nothing, nothing, true)
end

# Shared by `PhaseBuffers` and `simulate_phases`.
function _resolve_grid_step(atm_spec::AtmosphereSpec, plate_size, d, grid_step)
    grid_step !== nothing && return grid_step
    if d === nothing
        atm_spec isa SavedPhases || throw(ArgumentError(
            "Either aperture diameter `d` or `grid_step` must be provided."))
        return atm_spec.grid_step === nothing ? 1 : atm_spec.grid_step
    end
    return d / maximum(plate_size)
end

"""
    PhaseBuffers(atm_spec::AtmosphereSpec, plate_size[, d]; grid_step, batch, deviceadapter)

Everything needed to draw batches of phase screens. Owns no output array — allocate one with
[`allocate_phases`](@ref) and pass it to [`samplephases!`](@ref).

Groups a sampler ([`KarhunenLoeveSampler`](@ref) or [`SavedPhaseReader`](@ref)), an optional
[`HardingInterpolator`](@ref) with its low-resolution intermediate, and the ``r_0`` scale factor.

# Arguments
- `atm_spec`: an `AtmosphereSpec` describing the turbulence.
- `plate_size`: `(nx, ny)` size of the generated screens.
- `d`: aperture diameter in the same units as ``r_0``, giving `grid_step = d / maximum(plate_size)`.

# Keyword Arguments
- `grid_step`: physical size of one pixel; overrides the value derived from `d`.
- `batch`: screens per `samplephases!` call (default `$DEFAULT_BATCH`).
- `deviceadapter`: storage adapter or [`ComputeBackend`](@ref) for the buffers.
"""
struct PhaseBuffers{ST, IT, LT, FT}
    sampler::ST
    interpolator::IT
    low_array::LT
    phase_scale::FT
end
PhaseBuffers(atm_spec::AtmosphereSpec, plate_size::NTuple{2,Int}, d=nothing;
        grid_step=nothing, batch::Int=DEFAULT_BATCH, deviceadapter=Array) =
    prepare_phasebuffers(atm_spec, plate_size,
        _resolve_grid_step(atm_spec, plate_size, d, grid_step), batch, deviceadapter)

plate_size(pb::PhaseBuffers) = if isnothing(pb.interpolator)
    plate_size(pb.sampler)
else
    plate_size(pb.interpolator)
end
batch_length(pb::PhaseBuffers) = batch_length(pb.sampler)

"""
    samplephases!(out, pb::PhaseBuffers)

Draw the next batch of phase screens into `out`.
"""
function samplephases!(out, pb::PhaseBuffers)
    if pb.interpolator === nothing || nstages(pb.interpolator) == 0
        samplephases!(out, pb.sampler, pb.phase_scale)
    else
        samplephases!(pb.low_array, pb.sampler)
        harding_interpolate!(out, pb.interpolator, pb.low_array, 1, pb.phase_scale)
    end
end

"""
    allocate_phases(pb::PhaseBuffers[, deviceadapter])

Allocate a fresh `(plate_size(pb)..., batch_length(pb))` array to draw into. `PhaseBuffers` does
not own an output buffer, so the caller allocates — see [`samplephases!`](@ref).
"""
allocate_phases(pb::PhaseBuffers) =
    isnothing(pb.interpolator) || nstages(pb.interpolator) == 0 ?
        similar(allocate_phases(pb.sampler), (plate_size(pb)..., batch_length(pb))) :
        similar(pb.interpolator.work_bufs[1][1], (plate_size(pb)..., batch_length(pb)))
samplephases(phs_sampler) = samplephases!(allocate_phases(phs_sampler), phs_sampler)
