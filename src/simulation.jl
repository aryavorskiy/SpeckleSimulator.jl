function padded_plate_size(atm_spec::AtmosphereSpec, img_spec::ImagingSpec)
    if iszero(img_spec.exposure.exptime) || all(iszero, atm_spec.wind_velocity)
        return plate_size(img_spec)
    end
    max_offset = atm_spec.wind_velocity .* img_spec.exposure.exptime ./ img_spec.grid_step
    return plate_size(img_spec) .+ ceil.(Int, abs.(NoUnits.(max_offset)))
end

"""
    SimulationBuffers

Everything one simulation run needs, in one place: the phase sampler and its output block, and
(when imaging is enabled) the imaging scratch, the PSF block and the output image block.

Internal to the simulation loop -- the pieces are separately usable on their own, this only saves
threading five arguments through `simulation_run`. `image_buffers`, `psf` and `images` are
`nothing` for a phases-only run.
"""
struct SimulationBuffers{PB<:PhaseBuffers, PhT, OB, PlanT, PsfT, OutT}
    phase_buffers::PB
    phases::PhT
    optical_buffers::OB
    img_plan::PlanT
    psf::PsfT
    images::OutT
end
SimulationBuffers(pb::PhaseBuffers) =
    SimulationBuffers(pb, allocate_phases(pb), nothing, nothing, nothing, nothing)
function SimulationBuffers(::Type{T}, atm_spec, img_spec::ImagingSpec; batch::Int=1, deviceadapter=identity) where {T}
    pb = PhaseBuffers(atm_spec, padded_plate_size(atm_spec, img_spec); batch, deviceadapter, img_spec.grid_step)
    img_plan = adapt(deviceadapter, ImagingPlan(atm_spec, img_spec))
    ob = OpticalBuffers(img_plan; batch)
    SimulationBuffers(pb, allocate_phases(pb), ob, img_plan, imgbuf(ob), imgbuf(T, ob))
end
batch_length(sb::SimulationBuffers) = batch_length(sb.phase_buffers)

function simulation_run!!(img_bd::BufferedDataset, phs_bd::BufferedDataset,
        sb::SimulationBuffers, true_sky_adapt, n; verbose=true)
    batch = batch_length(sb)
    p = Progress(n, desc="Simulating images", enabled=verbose, dt=1)
    for j in 1:cld(n, batch)
        samplephases!(sb.phases, sb.phase_buffers)
        if sb.optical_buffers !== nothing
            compute_psf!(sb.psf, sb.optical_buffers, sb.img_plan, sb.phases)
            apply_truesky!(sb.psf, true_sky_adapt, sb.optical_buffers)
            readout!(sb.images, sb.psf, sb.img_plan.img_spec.readout)
            write_batch!(img_bd, j, sb.images)
        end
        write_batch!(phs_bd, j, sb.phases)
        next!(p, step=min(batch, n - (j - 1) * batch))
    end
    finish!(p)
end

function simulation_run(file, sb::SimulationBuffers, true_sky_adapt, n;
            verbose=true, savephases::Bool=true)
    open_file(file) do fid
        batch = batch_length(sb)
        if sb.images !== nothing
            img_bd = prepare_dataset(fid, "images", eltype(sb.images),
                size(sb.images)[1:2], n, batch)
        else
            img_bd = BufferedDataset(nothing)
        end
        if savephases
            phs_bd = prepare_dataset(fid, "phases", eltype(sb.phases),
                plate_size(sb.phase_buffers), n, batch)
        else
            phs_bd = BufferedDataset(nothing)
        end
        simulation_run!!(img_bd, phs_bd, sb, true_sky_adapt, n; verbose=verbose)
        _data(x) = x.dataset isa Array ? x.dataset : nothing
        if sb.images === nothing
            return _data(phs_bd)
        else
            out_ntuple = (phases = _data(phs_bd), images = _data(img_bd))
            if all(isnothing, out_ntuple)
                return nothing
            else
                return out_ntuple
            end
        end
    end
end

"""
    simulate_phases(atm_spec::AtmosphereSpec, plate_size[, d]; n, [grid_step, batch, file, verbose, deviceadapter])

Simulate `n` phase screens using the provided atmosphere specification and write
the results to an HDF5 file.

# Arguments
- `atm_spec`: an `AtmosphereSpec` used to produce phase screens.
- `plate_size`: the size of the phase screens to simulate.
- `d`: aperture diameter of the saved phase screens (in the same units as ``r_0``). When omitted,
  tries to infer it from `atm_spec`, throws if it is not a `SavedPhases`. Overriden by `grid_step`.

# Keyword Arguments
- `n`: number of phase screens to simulate.
- `grid_step`: physical size of one aperture pixel in the same units as ``r_0``. Overrides
  the value derived from `d` when provided.
- `batch`: batch size for buffered computations and HDF5 writes (default 128).
- `file`: output options. Can be a string (filename) or an `HDF5File` object. If set to `nothing`
    (default), no file is written and the phases are returned as an array.
- `verbose`: show progress meter (true by default).
- `deviceadapter`: adapter for device-backed arrays (defaults to `Array`). To use GPU arrays,
  pass e.g. `CUDA.CuArray` here (requires CUDA.jl).
"""
function simulate_phases(atm_spec::AtmosphereSpec, plate_size, d=nothing::Maybe{Number}; n::Int,
        grid_step::Maybe{Number}=nothing, batch::Int=DEFAULT_BATCH, file=nothing,
        verbose=true, deviceadapter=Array)
    batch = min(batch, n)
    phase_buffers = PhaseBuffers(atm_spec, plate_size, d;
        grid_step=grid_step, batch=batch, deviceadapter=deviceadapter)
    simulation_run(file, SimulationBuffers(phase_buffers), nothing, n; verbose=verbose)
end

"""
    simulate_images([T, true_sky::TrueSky, ]atm_spec::AtmosphereSpec, img_spec::ImagingSpec; \
        n, [batch, file, verbose, savephases, deviceadapter])

Simulate `n` images using the provided imaging and atmosphere specifications and write
the results to an HDF5 file.

# Arguments
- `T`: output image numeric type; if not provided, defaults to `Int` for finite-photon
    simulations (determined by `img_spec.readout_spec.nphotons`) and `Float64` for infinite-photon models.
- `true_sky`: a `TrueSky` model (e.g. `PointSource`, `DoubleSystem`, `TrueSkyImage`).
- `atm_spec`: an `AtmosphereSpec` used to produce phase screens.
- `img_spec`: an `ImagingSpec` describing the aperture, image size, photon budget, readout and filter.

# Keyword Arguments
- `n`: number of images to simulate.
- `batch`: batch size for buffered computations and HDF5 writes (default 128).
- `file`: output options. Can be a string (filename) or an `HDF5File` object. If set to `nothing`
    (default), no file is written and the images and phases are returned as a `NamedTuple` of arrays.
- `verbose`: show progress meter (true by default).
- `savephases`: when true (default), the sampled phase screens are saved in the HDF5 in dataset with
  key `"phases"`.
- `deviceadapter`: adapter for device-backed arrays. To use GPU
  arrays, pass e.g. `CUDA.CuArray` here (requires CUDA.jl).
"""
function simulate_images(::Type{T}, true_sky::TrueSky, atm_spec::AtmosphereSpec, img_spec::ImagingSpec;
    n::Int, batch::Int=DEFAULT_BATCH, file=nothing, verbose=true, savephases::Bool=true, deviceadapter=identity) where {T}
    if !isfinite_photons(img_spec.readout_spec) && T <: Integer
        throw(ArgumentError("Integer image eltype not compatible with infinite-photon imaging spec."))
    end
    if true_sky isa TrueSkyImage && size(true_sky.true_sky_fft) != image_size(img_spec)
        throw(ArgumentError("TrueSkyImage size $(size(true_sky.true_sky_fft)) does not match " *
            "image size $(image_size(img_spec))."))
    end

    batch = min(batch, n)
    true_sky_adapt = adapt(deviceadapter, true_sky)
    simulation_run(file, SimulationBuffers(T, atm_spec, img_spec; batch, deviceadapter), true_sky_adapt, n;
        verbose=verbose, savephases=savephases)
end
simulate_images(::Type{T}, atm_spec::AtmosphereSpec, img_spec::ImagingSpec; kwargs...) where {T} =
    simulate_images(T, PointSource(), atm_spec, img_spec; kwargs...)
simulate_images(true_sky::TrueSky, atm_spec::AtmosphereSpec, img_spec::ImagingSpec{T}; kwargs...) where {T} =
    simulate_images(isfinite_photons(img_spec.readout_spec) ? Int : T, true_sky, atm_spec, img_spec; kwargs...)
simulate_images(atm_spec::AtmosphereSpec, img_spec::ImagingSpec; kwargs...) =
    simulate_images(PointSource(), atm_spec, img_spec; kwargs...)
