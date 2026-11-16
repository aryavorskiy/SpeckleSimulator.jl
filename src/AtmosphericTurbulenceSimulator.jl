module AtmosphericTurbulenceSimulator

using Unitful
export @u_str
import Unitful: m, cm, mm, μm, nm
export m, cm, mm, μm, nm
import Unitful: s, ms, μs, ns
export s, ms, μs, ns

include("utils.jl")
export get_num_threads, set_num_threads
include("io.jl")
export HDF5File
include("atmosphere.jl")
export SavedPhases, SingleLayer, kolmogorov_covmat
# Low-level phase-generation API
export HardingInterpolator, KarhunenLoeveSampler, PhaseBuffers, SavedPhaseReader,
    harding_interpolate!, harding_upsample!, samplephases!, scale_phases!

include("imaging.jl")
export CircularAperture, DoubleSystem, Exposure, FilterSpec, ImagingSpec, PointSource,
    ReadoutSpec, TrueSkyImage
# Low-level imaging API
export BilinearInterpolator, ImagingPlan, OpticalBuffers, apply_truesky!, compute_images,
    compute_images!, compute_psf!, phases_to_pupil!, pupil_to_psf!, readout!
include("simulation.jl")
export simulate_images, simulate_phases

include("precompile.jl")

"""
    speckle_viewer(; kwargs...)

Open an interactive Makie window showing a simulated phase screen and the speckle image it
produces. Controls are split into two columns:

- **Atmosphere**: sliders for the Fried parameter ``r_0`` and the wind speed (direction fixed
  horizontal), plus a *New phase* button that draws a fresh random phase screen.
- **Imaging**: sliders for the central wavelength, filter bandwidth and exposure time.

The phase screen is recomputed only when an atmosphere control changes (or the button is pressed);
the speckle image is recomputed on any control change. A fixed circular aperture is used.

# Keyword Arguments
- `wavelength_range`: central-wavelength slider values (Unitful length; default nm).
- `bw_range`: filter-bandwidth slider values (Unitful length; default nm).
- `exptime_range`: exposure-time slider values (Unitful time; default s).
- `r0_range`: Fried-parameter slider values (Unitful length; default cm).
- `wind_range`: wind-speed slider values (Unitful velocity; default cm/s).
- `d`: aperture diameter (Unitful length; default `2m`).
- `aperture`: aperture array (defaults to 64×64 circular aperture).
"""
function speckle_viewer end
export speckle_viewer

end # module AtmosphericTurbulenceSimulator
