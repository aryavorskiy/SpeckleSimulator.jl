using LinearAlgebra, FFTW, Distributions, HDF5, ProgressMeter, ChunkSplitters
import Adapt: adapt, adapt_storage, adapt_structure

"""
    FilterSpec

Representation of a spectral filter used by the imaging pipeline.

---
    FilterSpec(wavelengths[, intensities])

# Arguments
- `wavelengths`: vector of sampled wavelengths within the filter bandpass. Should be a Unitful
  length; plain numbers are assumed to be nanometers.
- `intensities`: vector of relative intensities at each sampled wavelength. If not provided,
  equal weights are assumed.
"""
struct FilterSpec{T1,T2}
    wavelengths::Vector{T1}
    intensities::Vector{T2}
    FilterSpec{T1,T2}(wavelengths, intensities) where {T1,T2} = new{T1,T2}(wavelengths, intensities)
end
function FilterSpec(wavelengths::AbstractVector, intensities::AbstractVector=ones(Int, length(wavelengths)))
    wl = _as_wavelength.(wavelengths)
    return FilterSpec{eltype(wl),eltype(intensities)}(wl, intensities)
end
nwavel(fs::FilterSpec) = length(fs.wavelengths)

"""
    FilterSpec(base_wavelength; [bandwidth=0, tcenter=1, tedge=1, npts=7])

# Arguments
- `base_wavelength`: central wavelength for the filter. Should be a Unitful length; a plain
  number is assumed to be nanometers.

# Keyword Arguments
- `bandwidth`: total width of the filter bandpass in wavelength units. If zero, the filter is monochromatic.
- `tcenter`: relative intensity at the center wavelength (default 1).
- `tedge`: relative intensity at the edges of the bandpass (default 1).
- `npts`: number of sample points across the bandpass (default 7).
"""
function FilterSpec(base_wavelength::Number=DEFAULT_WAVELEN; bandwidth=0, tcenter=1, tedge=1, npts=7)
    base = _as_wavelength(base_wavelength)
    iszero(bandwidth) && return FilterSpec([base], [tcenter])
    bw_units = _as_wavelength(bandwidth)
    wavelengths = range(base - bw_units / 2, base + bw_units / 2, length=npts)
    intensities = range(-pi/2, pi/2, length=npts) .|> x -> cos(x) * (tcenter - tedge) + tedge
    return FilterSpec(wavelengths, intensities)
end

struct BilinearInterpolator{IT,VT}
    ix::IT
    iy::IT
    ixp1::IT
    iyp1::IT
    tx::VT
    ty::VT
    can_ff::Bool
end
function BilinearScale(to::AbstractArray, scale::Real, sz=(size(to, 1), size(to, 2)))
    nx, ny = sz
    sx = range((1 - nx ÷ 2) / scale + nx ÷ 2 + 1, step=1/scale, length=nx)
    sy = range((1 - ny ÷ 2) / scale + ny ÷ 2 + 1, step=1/scale, length=ny)
    ix = copy!(similar(to, Int, nx), clamp.(floor.(Int, sx), 1, nx))
    iy = copy!(similar(to, Int, ny), clamp.(floor.(Int, sy), 1, ny))
    ixp1 = copy!(similar(to, Int, nx), clamp.(ceil.(Int, sx), 1, nx))
    iyp1 = copy!(similar(to, Int, ny), clamp.(ceil.(Int, sy), 1, ny))
    tx = real(oftype(vec(to), sx) .- ix)
    ty = real(oftype(vec(to), sy) .- iy)
    return BilinearInterpolator(ix, iy, ixp1, iyp1, tx, ty, scale≈1)
end
function BilinearShift(to::AbstractArray, offset::NTuple{2}, sz=(size(to, 1), size(to, 2)))
    nx, ny = sz
    ix = range(floor(Int, 1 + offset[1]), length=nx)
    iy = range(floor(Int, 1 + offset[2]), length=ny)
    ixp1 = range(ceil(Int, 1 + offset[1]), length=nx)
    iyp1 = range(ceil(Int, 1 + offset[2]), length=ny)
    tx = convert(real(eltype(to)), offset[1] - floor(offset[1]))
    ty = convert(real(eltype(to)), offset[2] - floor(offset[2]))
    return BilinearInterpolator(ix, iy, ixp1, iyp1, tx, ty, all(isinteger, offset))
end
"""
    bilinear_interpolate!(to, from, interp[, α, β, g, h])

Performs bilinear interpolation of `from` onto `to` using the precomputed `interp` indices and weights.

# Arguments
- `to`: output array to be filled with interpolated values.
- `from`: input array to be interpolated.
- `interp`: a `BilinearInterpolator` object containing precomputed indices and weights.
- `α` and `β`: oprtional scaling factors for the output. If provided, the output is computed
  as `to = α * interpolate(from) + β * to`. Can be scalar or array of the same size as `to`.
- `g` and `h`: optional functions applied to the input after and before interpolation, respectively.
"""
function bilinear_interpolate!(to::AbstractArray, from::AbstractArray, interp::BilinearInterpolator,
        α=true, β=false, g=identity, h=identity)
    tx = interp.tx
    ty = transpose(interp.ty)

    if interp.can_ff
        @views @inbounds @. to = α * g(h(from[interp.ix, interp.iy, :])) + β * to
    else
        @views @inbounds @. to = α * g(
            (1 - tx) * (1 - ty) * h(from[interp.ix, interp.iy, :]) +
            tx * (1 - ty) * h(from[interp.ixp1, interp.iy, :]) +
            (1 - tx) * ty * h(from[interp.ix, interp.iyp1, :]) +
            tx * ty * h(from[interp.ixp1, interp.iyp1, :])) + β * to
    end
    return to
end

"""
    ReadoutSpec(nphotons[, background, readout, bias, gain])

Specifies the photon budget and readout process for imaging simulations.

# Arguments
- `nphotons`: total expected number of photons in the image. Set to `Inf` for continuous flux
  (no noise is applied and `background`, `readout`, `bias` and `gain` are ignored).
- `background`: expected number of background photons per pixel. Can be omitted if `nphotons` is `Inf`;
  required otherwise.
- `readout`: standard deviation of the Gaussian readout noise, in photoelectrons (before `gain`).
  Defaults to `0` (no readout noise).
- `bias`: constant offset added to every pixel after applying `gain`. Defaults to `0`.
- `gain`: multiplicative factor converting photoelectrons to output counts. Must be nonzero.
  Defaults to `1`.

Each pixel is computed as `gain * (Poisson(λ) + readout * randn()) + bias`, where
`λ = nphotons * intensity + background`, and then rounded to the output type.
"""
struct ReadoutSpec{T<:Real}
    nphotons::T
    background::T
    readout::T
    bias::T
    gain::T
end
function ReadoutSpec(nph, bg=nothing, readout=0, bias=0, gain=1)
    iszero(gain) && throw(ArgumentError("Zero gain is disallowed"))
    if isinf(nph)
        promoted_args = promote(nph, zero(nph), readout, bias, gain)
    else
        isnothing(bg) && throw(ArgumentError("Must specify background when `nphotons` is finite"))
        promoted_args = promote(nph, bg, readout, bias, gain)
    end
    return ReadoutSpec{eltype(promoted_args)}(promoted_args...)
end
convert_numtype(::Type{T}, pc::ReadoutSpec) where {T} =
    ReadoutSpec(convert.(T, (pc.nphotons, pc.background, pc.readout, pc.bias, pc.gain))...)
isfinite_photons(pc::ReadoutSpec) = isfinite(pc.nphotons)

const GAUSSIAN_POISSON_CUTOFF = 15
const KNUTH_MAXITER = round(Int, GAUSSIAN_POISSON_CUTOFF + 5 * sqrt(GAUSSIAN_POISSON_CUTOFF))
function _sample_poisson(::Type{T}, λ) where {T}
    if λ < GAUSSIAN_POISSON_CUTOFF
        # Knuth counting algorithm
        L = exp(-λ)
        k = 0
        p = float(one(λ))
        for _ in 1:KNUTH_MAXITER
            p *= rand(float(typeof(λ)))
            p <= L && break
            k += 1
        end
        return convert(T, k)
    end
    i3sl = 1/3sqrt(λ)
    return round(T, max(λ * (1 + i3sl * randn(typeof(λ)) - i3sl^2)^3, zero(λ)))
end
function readout!(dst::AbstractArray{T}, img::AbstractArray, pc::ReadoutSpec{FT}) where {T,FT}
    if isfinite_photons(pc)
        if iszero(pc.readout)
            @. dst = round(T, _sample_poisson(T, real(img) * pc.nphotons + pc.background) * pc.gain + pc.bias)
        else
            @. dst = round(T, (_sample_poisson(T, real(img) * pc.nphotons + pc.background) +
                    randn(FT) * pc.readout) * pc.gain + pc.bias)
        end
    else
        @. dst = img
    end
end

struct Exposure{ET<:Number}
    exptime::ET
    nsteps::Int
    round_offsets::Bool
end
"""
    Exposure(exptime[, nsteps][; round_offsets])

A struct that encapsulates the exposure time and related parameters for long-exposure simulations.
The exposure is simulated by averaging `nsteps` short exposures with appropriate offsets of the wavefront.

# Arguments
- `exptime`: total exposure time. Shares the same time units as the wind velocity in the atmosphere specification.
- `nsteps`: number of steps to simulate for long exposures (default 5).
- `round_offsets`: whether to round the exposure offsets to integers (default false). When true,
  the phase screens are sampled at integer pixel offsets, which can reduce interpolation artifacts.
"""
function Exposure(exptime, nsteps=5; round_offsets=false)
    nsteps == 1 && !iszero(exptime) &&
        @warn "Ignoring non-zero exposure time for single-step exposure."
    if iszero(exptime)
        nsteps = 1
    end
    Exposure(exptime, nsteps, round_offsets)
end

abstract type TrueSky end

"""
    PointSource()

Simple true-sky brightness model. Represents a non-resolved point source.
"""
struct PointSource <: TrueSky end

"""
    DoubleSystem(rel_position, intensity)

Model for a two-component source (binary): primary plus a secondary offset by `rel_position`.

# Arguments
- `rel_position`: `(dx, dy)` integer tuple specifying the secondary's pixel offset.
- `intensity`: multiplicative intensity of the secondary relative to the primary.
"""
struct DoubleSystem{T} <: TrueSky
    rel_position::NTuple{2,Int}
    intensity::T
    DoubleSystem(position, intensity::Real) =
        new{typeof(intensity)}(Tuple(position), intensity)
end

"""
    TrueSkyImage(true_sky::AbstractMatrix{T})

Wrap a real-valued true-sky image for use with the imaging pipeline. The pixel scale matches
that of the `ImagingSpec`.

# Arguments
- `true_sky`: real image array representing spatial sky brightness.
"""
struct TrueSkyImage{MT<:AbstractMatrix{<:Complex}} <: TrueSky
    true_sky_fft::MT
end
function TrueSkyImage(true_sky::AbstractMatrix{T}) where {T<:Real}
    true_sky_fft = fft(ifftshift(true_sky))
    true_sky_fft ./= true_sky_fft[1, 1]  # normalize DC component to 1
    return TrueSkyImage{typeof(true_sky_fft)}(true_sky_fft)
end
adapt_structure(to, ts::TrueSkyImage) = TrueSkyImage(adapt_storage(to, ts.true_sky_fft))

const FilterSpecType{T} = FilterSpec{<:Union{T, Quantity{T}}, <:Union{T, Quantity{T}}}
"""
    ImagingSpec

Container for the imaging system configuration. It is defined by the telescope `aperture`, the
source brightness via `readout_spec`, an optional spectral `filter_spec`, the `exposure`,
and the output `img_size`.
If `img_size` does not match the aperture’s Nyquist grid, the aperture is zero-padded accordingly.
"""
struct ImagingSpec{T, T2, AT<:AbstractMatrix{T}, FST<:FilterSpec}
    aperture::AT
    grid_step::T2
    readout_spec::ReadoutSpec{T}
    filter_spec::FST
    exposure::Exposure
    img_size::NTuple{2,Int}
    function ImagingSpec(
        aperture::AbstractMatrix{T},
        grid_step::Number,
        readout_spec::ReadoutSpec{T},
        filter_spec::FilterSpecType{T},
        exposure::Exposure,
        img_size::NTuple{2,Int}) where {T<:Real}
        new{T, typeof(grid_step), typeof(aperture), typeof(filter_spec)}(
            aperture, grid_step, readout_spec, filter_spec, exposure, img_size)
    end
end

"""
    ImagingSpec([T, ]aperture, d, [readout_spec][; grid_step, filter, nyquist_oversample, img_size])

Create an imaging system specification.

# Arguments
- `T`: desired number type, inferred from `aperture` if not provided.
- `aperture`: 2D aperture (pupil) array describing the telescope pupil.
- `d`: aperture diameter in the same units as ``r_0`` (e.g. `2m`). Used to compute `grid_step` as
  `d / maximum(size(aperture))`.
- `readout_spec`: `ReadoutSpec` instance describing the readout. Default is no readout.

# Keyword Arguments
- `filter`: `FilterSpec` describing sampled wavelengths and their relative intensities.
  Defaults to a monochromatic filter at 550 nm.
- `exposure`: a number or an [`Exposure`](@ref) instance describing the exposure time and number of
  steps for long exposures. Defaults to zero exposure time (i.e. short exposure).
- `nyquist_oversample`: multiplicative factor applied to the default Nyquist image size (`2 * size(aperture)`).
  Defaults to 1. Ignored if `img_size` is provided.
- `img_size`: explicit output image size `(nx, ny)`. If not provided, computed from aperture size
  and `nyquist_oversample`.
"""
function ImagingSpec(aperture::AbstractMatrix{T}, d::Number, readout_spec::ReadoutSpec=ReadoutSpec(Inf);
    filter::FilterSpec=FilterSpec(DEFAULT_WAVELEN),
    nyquist_oversample::Real=1, exposure::Union{Exposure,Number}=0,
    img_size::NTuple{2,Int}=round.(Int, size(aperture) .* 2 .* nyquist_oversample),
    grid_step::Number=d / maximum(size(aperture))) where T<:Real
    pc = convert_numtype(T, readout_spec)
    ex = exposure isa Number ? Exposure(exposure) : exposure
    ft = FilterSpec(
        convert_numtype(T, filter.wavelengths), convert_numtype(T, filter.intensities))
    return ImagingSpec(aperture, grid_step, pc, ft, ex, img_size)
end
ImagingSpec(::Type{T}, aperture::AbstractMatrix, args...; kw...) where T<:Real =
    ImagingSpec(convert.(T, aperture), args...; kw...)

adapt_structure(to, img_spec::ImagingSpec) =
    ImagingSpec(adapt_storage(to, img_spec.aperture), img_spec.grid_step,
    img_spec.readout_spec, img_spec.filter_spec, img_spec.exposure, img_spec.img_size)
plate_size(img_spec::ImagingSpec) = size(img_spec.aperture)
image_size(img_spec::ImagingSpec) = img_spec.img_size
psf_norm(img_spec::ImagingSpec) = sum(abs2, img_spec.aperture) * prod(img_spec.img_size) *
        sum(img_spec.filter_spec.intensities)

"""
    ImagingPlan(atm_spec::AtmosphereSpec, img_spec::ImagingSpec)

Everything the optical pipeline needs that does not change during a run: the `ImagingSpec`, the
atmosphere's `base_wavelength`, the per-exposure-step wind `offsets`, one [`FilterChannel`](@ref)
per sampled wavelength, and `psf_norm`.

Holds no scratch and no output, so it is shared read-only by every worker -- the per-wavelength
interpolation tables exist once per run rather than once per thread. Built on the host; `adapt` it
to a device when constructing [`ImageBuffers`](@ref), not here. `length(offsets)` is folded into
`psf_norm`, so a standalone [`readout!`](@ref) needs no extra divisor.
"""
struct ImagingPlan{ST<:ImagingSpec, WT, OT, CT, FT<:Real}
    spec::ST
    base_wavelength::WT
    offsets::Vector{OT}
    channels::Vector{CT}
    psf_norm::FT
end
function ImagingPlan(atm_spec::AtmosphereSpec, img_spec::ImagingSpec)
    offsets = long_exp_offsets(atm_spec, img_spec)
    NT = eltype(img_spec.aperture)
    filter = img_spec.filter_spec
    factors = NoUnits.(filter.wavelengths ./ atm_spec.base_wavelength)
    channels = map(eachindex(factors)) do w
        FilterChannel(img_spec.aperture, convert(NT, factors[w]),
            convert(NT, filter.intensities[w]), img_spec.img_size)
    end
    return ImagingPlan(img_spec, atm_spec.base_wavelength, offsets, channels,
        psf_norm(img_spec) * length(offsets))
end
adapt_structure(to, plan::ImagingPlan) = ImagingPlan(adapt(to, plan.spec),
    plan.base_wavelength, plan.offsets, map(c -> adapt(to, c), plan.channels), plan.psf_norm)
image_size(plan::ImagingPlan) = image_size(plan.spec)
plate_size(plan::ImagingPlan) = plate_size(plan.spec)

"""
    FilterChannel(interpolator, phs_factor, weight)

One sampled wavelength of the filter bandpass, bundling the three quantities derived from it:
the plate-scale `interpolator` ([`BilinearScale`](@ref)), the phase scaling `phs_factor`
(``λ_w / λ_{base}``) and the intensity `weight`. They are carried together because
all three come from the same wavelength -- keeping them apart meant recomputing the weight in the
innermost loop.
"""
struct FilterChannel{IT<:BilinearInterpolator, T}
    interpolator::IT
    phs_factor::T
    weight::T
end
adapt_structure(to, interp::BilinearInterpolator) = BilinearInterpolator(
    adapt(to, interp.ix), adapt(to, interp.iy), adapt(to, interp.ixp1), adapt(to, interp.iyp1),
    adapt(to, interp.tx), adapt(to, interp.ty), interp.can_ff)
adapt_structure(to, ch::FilterChannel) =
    FilterChannel(adapt(to, ch.interpolator), ch.phs_factor, ch.weight)
FilterChannel(aperture::AbstractArray, lambda_rel, inten, imgsize = (size(aperture, 1), size(aperture, 2))) =
    FilterChannel(BilinearScale(aperture, lambda_rel, imgsize), 1 / lambda_rel, inten / lambda_rel^2)

"""
    OpticalBuffers(plan::ImagingPlan, batch)
    OpticalBuffers(pupil_buffer)

Per-worker scratch for the optical pipeline: two complex `(img_size..., batch)` buffers and the
FFT plan over them. Nothing else -- the per-wavelength constants live in [`ImagingPlan`](@ref) and
the PSF accumulator and readout image are supplied by the caller, see [`compute_psf!`](@ref) and
[`readout!`](@ref).
"""
struct OpticalBuffers{MT, PT}
    pupil_buffer::MT
    focal_buffer::MT
    fftplan::PT
end
image_size(bufs::OpticalBuffers) = size(bufs.pupil_buffer)[1:2]
batch_length(bufs::OpticalBuffers) = size(bufs.pupil_buffer, 3)

OpticalBuffers(pupil_buffer::AbstractArray{<:Complex, 3}) =
    OpticalBuffers(pupil_buffer, similar(pupil_buffer), plan_fft(pupil_buffer, (1, 2)))
OpticalBuffers(img_array::AbstractArray, batch::Int=DEFAULT_BATCH, sz::Tuple=size(img_array)[1:2]) =
    OpticalBuffers(similar(img_array, complex(eltype(img_array)), sz..., batch))
OpticalBuffers(plan::ImagingPlan, batch::Int=DEFAULT_BATCH) =
    OpticalBuffers(plan.spec.aperture, batch, plan.spec.img_size)

"""
    phases_to_pupil!(pupil, phases, aperture, offset, phs_factor)

Write the complex pupil field `aperture * cis(phase * phs_factor)` into `pupil`, zero-padded to
the image grid and sampled at the wind `offset`.
"""
function phases_to_pupil!(pupil::AbstractArray, phases, aperture, offset::Tuple=(0, 0), phs_factor=true)
    M, N = size(aperture)
    Cx, Cy = size(pupil) .÷ 2
    fill!(pupil, 0)
    ap_slice = @view pupil[Cx - M ÷ 2 + 1:Cx - M ÷ 2 + M, Cy - N ÷ 2 + 1:Cy - N ÷ 2 + N, :]
    bilinear_interpolate!(ap_slice, phases, BilinearShift(aperture, offset), aperture, true,
        cis, Base.Fix2(*, phs_factor))
    return pupil
end
phases_to_pupil!(buf::OpticalBuffers, args...) = phases_to_pupil!(buf.pupil_buffer, args...)

"""
    pupil_to_psf!(psf, bufs::OpticalBuffers, ch::FilterChannel[, α, β])

Propagate the pupil field currently in `bufs` to the focal plane and **accumulate** its intensity
into `psf`, rescaled to `ch`'s plate scale and weighted by `α * ch.weight`. Does not zero `psf`.
"""
function pupil_to_psf!(psf, bufs::OpticalBuffers, ch::FilterChannel, α=true, β=false)
    mul!(bufs.focal_buffer, bufs.fftplan, bufs.pupil_buffer)
    fftshift!(bufs.pupil_buffer, bufs.focal_buffer, (1, 2))
    bilinear_interpolate!(psf, bufs.pupil_buffer, ch.interpolator, α * ch.weight, β, identity, abs2)
    return psf
end

"""
    compute_psf!(psf, bufs::OpticalBuffers, plan::ImagingPlan, phases[, α, β])

Accumulate the (unnormalised) point-spread function over every exposure offset and wavelength:
`psf = α * PSF(phases) + β * psf`. Divide by `plan.psf_norm` for a flux-normalised PSF, or hand
it to [`readout!`](@ref).
"""
function compute_psf!(psf, bufs::OpticalBuffers, plan::ImagingPlan, phases, α=true, β=false)
    firstiter = true
    for offset in plan.offsets, ch in plan.channels
        phases_to_pupil!(bufs, phases, plan.spec.aperture, offset, ch.phs_factor)
        pupil_to_psf!(psf, bufs, ch, α / plan.psf_norm, firstiter ? β : one(β))
        firstiter = false
    end
    return psf
end

"""
    apply_truesky!(psf, true_sky[, bufs::OpticalBuffers])

Convolve `psf` in place with a [`TrueSky`](@ref) brightness model, using `bufs` as scratch.
"""
function apply_truesky!(psf, ts::TrueSkyImage, opt_buffer::OpticalBuffers=OpticalBuffers(psf))
    copyto!(opt_buffer.pupil_buffer, psf)
    mul!(opt_buffer.focal_buffer, opt_buffer.fftplan, opt_buffer.pupil_buffer)
    opt_buffer.focal_buffer .*= ts.true_sky_fft
    ldiv!(opt_buffer.pupil_buffer, opt_buffer.fftplan, opt_buffer.focal_buffer)
    psf .= real.(opt_buffer.pupil_buffer)
    return psf
end
function apply_truesky!(psf, ds::DoubleSystem)
    img = psf
    @assert all(abs.(ds.rel_position) .< size(psf)[1:2] .÷ 2)
    o1, o2 = ds.rel_position
    s1_dest, s1_src = o1 > 0 ? (o1 + 1:size(img, 1), 1:size(img, 1) - o1) : (1:size(img, 1) + o1, -o1 + 1:size(img, 1))
    s2_dest, s2_src = o2 > 0 ? (o2 + 1:size(img, 2), 1:size(img, 2) - o2) : (1:size(img, 2) + o2, -o2 + 1:size(img, 2))
    @views @. img[s1_dest, s2_dest, :] += img[s1_src, s2_src, :] * ds.intensity
    img ./= (1 + ds.intensity)
    return psf
end
apply_truesky!(psf, ::PointSource) = psf
apply_truesky!(psf, ts, ::OpticalBuffers) = apply_truesky!(psf, ts)

"""
    CircularAperture([T, ]sz, [radius; aa_dist=1])

Create a circular (optionally anti-aliased) aperture array of shape `sz`. Returns a 2D
numeric array suitable for use as an aperture in `ImagingSpec`.

# Arguments
- `T`: desired number type, `Float64` by default.
- `sz`: aperture size `(nx, ny)`.
- `radius`: radius of the circular aperture in pixels. Defaults to the largest that fits.
- `aa_dist`: anti-aliasing transition width in pixels at the aperture edge.
"""
function CircularAperture(::Type{T}, sz::NTuple{2}, radius=minimum((sz .- 1) .÷ 2); aa_dist=1) where T<:Real
    aperture = zeros(T, sz)
    X, Y = sz .÷ 2 .+ 1
    for I in eachindex(IndexCartesian(), aperture)
        x, y = I[1] - X, I[2] - Y
        r = sqrt(x^2 + y^2)
        if r < radius - aa_dist / 2
            aperture[I] = 1
        elseif r > radius + aa_dist / 2
            aperture[I] = 0
        else
            aperture[I] = 0.5 - (r - radius) / aa_dist
        end
    end
    return aperture
end
CircularAperture(sz::NTuple{2}, radius=minimum((sz .- 1) .÷ 2); kw...) =
    CircularAperture(Float64, sz, radius; kw...)

function padded_plate_size(atm_spec::AtmosphereSpec, img_spec::ImagingSpec)
    if iszero(img_spec.exposure.exptime) || all(iszero, atm_spec.wind_velocity)
        return plate_size(img_spec)
    end
    max_offset = atm_spec.wind_velocity .* img_spec.exposure.exptime ./ img_spec.grid_step
    return plate_size(img_spec) .+ ceil.(Int, abs.(NoUnits.(max_offset)))
end
function long_exp_offsets(atm_spec::AtmosphereSpec, img_spec::ImagingSpec)
    n = img_spec.exposure.nsteps
    if n == 1 || iszero(img_spec.exposure.exptime) || all(iszero, atm_spec.wind_velocity)
        offset_list = [ustrip.(atm_spec.wind_velocity .* img_spec.exposure.exptime .* 0)]
    else
        offset_list = [NoUnits.(atm_spec.wind_velocity .* (img_spec.exposure.exptime * j / (n - 1) / img_spec.grid_step)) for j in 0:n-1]
    end
    if img_spec.exposure.round_offsets
        offset_list = [round.(offset) for offset in offset_list]
    end
    mins = minimum(first, offset_list), minimum(last, offset_list)
    return [offset .- mins for offset in offset_list]
end

"""
    ImageBuffers(plan::ImagingPlan, batch[, deviceadapter])

Scratch for a batched imaging run: one [`OpticalBuffers`](@ref) per worker, the batch chunk ranges
they cover, and the device-adapted [`ImagingPlan`](@ref) they share.
"""
struct ImageBuffers{BT<:OpticalBuffers, PLT<:ImagingPlan, CT}
    opt_bufs::Vector{BT}
    chunk_ranges::Vector{CT}
    plan::PLT
end
image_size(ib::ImageBuffers) = image_size(ib.plan)
batch_length(ib::ImageBuffers) = sum(length, ib.chunk_ranges)

allocate_images(::Type{T}, ib::ImageBuffers) where {T} =
    similar(ib.plan.spec.aperture, T, image_size(ib)..., batch_length(ib))
allocate_images(ib::ImageBuffers) =
    allocate_images(real(float(eltype(ib.plan.spec.aperture))), ib)

function ImageBuffers(plan::ImagingPlan, batch::Int,
        adapter=identity)
    chunk_ranges = collect(chunks(1:batch; n=get_num_threads()))
    plan_adapt = adapt(adapter, plan)
    opt_bufs = map(cr -> OpticalBuffers(plan_adapt, length(cr)), chunk_ranges)
    return ImageBuffers(opt_bufs, chunk_ranges, plan_adapt)
end
ImageBuffers(spec::ImagingSpec, atm_spec::AtmosphereSpec, args...) =
    ImageBuffers(ImagingPlan(atm_spec, spec), args...)

function prepare_buffers(atm_spec, img_spec::ImagingSpec, batch::Int, adapter)
    return prepare_phasebuffers(atm_spec, padded_plate_size(atm_spec, img_spec),
            img_spec.grid_step, batch, adapter),
        ImageBuffers(ImagingPlan(atm_spec, img_spec), batch, adapter)
end

"""
    compute_psf!(dst, bufs::ImageBuffers, phases[, α, β])

Performs `compute_psf!` in parallel over the batch dimension.
"""
function compute_psf!(dst, bufs::ImageBuffers, phases, α=true, β=false)
    if length(bufs.opt_bufs) == 1
        compute_psf!(dst, bufs.opt_bufs[1], bufs.plan, phases, α, β)
    else
        @views Threads.@threads for j in eachindex(bufs.opt_bufs)
            cr = bufs.chunk_ranges[j]
            compute_psf!(dst[:, :, cr], bufs.opt_bufs[j], bufs.plan, phases[:, :, cr], α, β)
        end
    end
end

"""
    apply_truesky!(dst, ts::TrueSky, bufs::ImageBuffers)

Performs `apply_truesky!` in parallel over the batch dimension.
"""
function apply_truesky!(dst, ts::TrueSky, bufs::ImageBuffers)
    if length(bufs.opt_bufs) == 1
        apply_truesky!(dst, ts, only(bufs.opt_bufs))
    else
        @views Threads.@threads for j in eachindex(bufs.opt_bufs)
            cr = bufs.chunk_ranges[j]
            apply_truesky!(dst[:, :, cr], ts, bufs.opt_bufs[j])
        end
    end
end

"""
    readout!(dst, img, pc::ReadoutSpec, zchunks)

Performs `readout!` in parallel over the batch dimension.
"""
function readout!(dst, img, pc::ReadoutSpec, zchunks)
    if length(zchunks) == 1
        readout!(dst, img, pc)
    else
        @views Threads.@threads for zchunk in zchunks
            readout!(dst[:, :, zchunk], img[:, :, zchunk], pc)
        end
    end
end
readout!(dst, img, bufs::ImageBuffers) =
    readout!(dst, img, bufs.plan.spec.readout_spec, bufs.chunk_ranges)
