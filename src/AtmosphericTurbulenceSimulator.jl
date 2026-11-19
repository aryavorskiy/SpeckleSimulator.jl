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

function speckle_viewer end
export speckle_viewer

end # module AtmosphericTurbulenceSimulator
