import AtmosphericTurbulenceSimulator: isfinite_photons, convert_numtype,
    nstages, plate_size
using Unitful

@testset "Constructors" begin
    @testset "SingleLayer" begin
        atm = SingleLayer(5.0)
        @test atm isa SingleLayer
        @test atm.r₀ == 5.0
        @test atm.base_wavelength == 550nm

        atm = SingleLayer(0.5m; base_wavelength=500nm, wind_velocity=(1m/s, 1m/s))
        @test atm.r₀ == 0.5m
        @test atm.base_wavelength == 500nm
        @test atm.wind_velocity == (1m/s, 1m/s)

        atm32 = SingleLayer(Float32, 0.5m)
        @test atm32.r₀ isa Quantity{Float32}
        @test unit(atm32.r₀) == m

        # Test Harding interpolation
        hi = HardingInterpolator((64, 64); batch=4, nsteps=:auto)
        @test nstages(hi) == 2
        @test plate_size(hi) == (64, 64)
        @test first(hi.stage_sizes) == (25, 25)

        hi_1pass = HardingInterpolator((64, 64); batch=4, nsteps=1)
        @test nstages(hi_1pass) == 1
        @test plate_size(hi_1pass) == (64, 64)
        @test first(hi_1pass.stage_sizes) == (38, 38)

        # Test that small size_from throws error
        @test_throws ArgumentError HardingInterpolator((64, 64); batch=4, size_from=(8, 8))
    end

    @testset "SavedPhases" begin
        tmpfile = tempname() * ".h5"
        data = reshape(Float32.(1:4*5*6), 4, 5, 6)
        h5write(tmpfile, "phases", data)

        h5open(tmpfile, "r") do fid
            atm = SavedPhases(fid["phases"]; wind_velocity=(1, 2))
            @test atm isa SavedPhases
            @test atm.wind_velocity == (1, 2)

            atm2 = SavedPhases(fid["phases"])
            @test atm2.wind_velocity == (0, 0)
        end
        rm(tmpfile, force=true)
    end

    @testset "FilterSpec" begin
        filter = FilterSpec(500; bandwidth=100, npts=5)
        @test filter.wavelengths == range(450, 550, length=5) * nm
        @test filter.intensities == ones(5)

        filter2 = FilterSpec(0.5μm; bandwidth=0.1μm, npts=5)
        @test filter2.wavelengths ≈ range(450, 550, length=5) * nm
        @test filter2.intensities == ones(5)

        filter_mono = FilterSpec(600, tcenter=0.8)
        @test filter_mono.wavelengths == [600nm]
        @test filter_mono.intensities == [0.8]

        # Non-flat intensities (tcenter ≠ tedge)
        filter_shaped = FilterSpec(550.0; bandwidth=100.0, tcenter=1.0, tedge=0.5, npts=7)
        @test filter_shaped.intensities[4] ≈ 1.0          # center
        @test filter_shaped.intensities[1] ≈ 0.5          # edges
        @test filter_shaped.intensities[end] ≈ 0.5

        # Direct vector constructor
        wl = [480.0, 550.0, 620.0]
        intens = [0.8, 1.0, 0.8]
        filter_vec = FilterSpec(wl, intens)
        @test filter_vec.wavelengths == wl * nm
        @test filter_vec.intensities == intens
    end

    @testset "CircularAperture" begin
        ap = CircularAperture((32, 32), 15)
        @test size(ap) == (32, 32)
        @test all(0 .<= ap .<= 1)

        # center is filled
        @test ap[16, 16] ≈ 1.0
        # corners are empty
        @test ap[1, 1] ≈ 0.0
        @test ap[1, 32] ≈ 0.0
        @test ap[32, 1] ≈ 0.0
        @test ap[32, 32] ≈ 0.0

        # Test normalization (sum should equal area of circle, approximately)
        expected_pixels = π * 15^2
        @test sum(ap) ≈ expected_pixels rtol=0.05
    end

    @testset "ImagingSpec" begin
        ap = CircularAperture((32, 32), 15)
        img_spec = ImagingSpec(ap, 32, ReadoutSpec(Inf))
        @test size(img_spec.aperture) == (32, 32)
        @test img_spec.img_size == (64, 64)
        @test img_spec.grid_step == 1

        img_spec_custom = ImagingSpec(ap, 32, ReadoutSpec(Inf); nyquist_oversample=1.5)
        @test img_spec_custom.img_size == (96, 96)

        img_spec = ImagingSpec(ap, 1m, ReadoutSpec(Inf))
        @test img_spec.grid_step == 1m / size(ap, 1)

        # Test with filter
        img_spec_filter = ImagingSpec(ap, 32, ReadoutSpec(1e6, 10); filter=FilterSpec(500.0; bandwidth=100.0))
        @test img_spec_filter.filter_spec.wavelengths[end÷2+1] == 500nm
        @test img_spec_filter.readout_spec.nphotons == 1e6
        @test img_spec_filter.readout_spec.background == 10

        @test_throws ArgumentError ImagingSpec(ap, ReadoutSpec(1e6))
    end

    @testset "ReadoutSpec" begin
        pc_inf = ReadoutSpec(Inf)
        @test !isfinite_photons(pc_inf)
        @test pc_inf.background == 0.0

        pc = ReadoutSpec(1e6, 100.0)
        @test isfinite_photons(pc)
        @test convert_numtype(Float32, pc).nphotons ≈ Float32(1e6)

        # Finite nphotons without background must throw
        @test_throws ArgumentError ReadoutSpec(1e6)
    end

    @testset "Exposure" begin
        ex = Exposure(2.0, 5)
        @test ex.exptime == 2.0
        @test ex.nsteps == 5
        @test ex.round_offsets == false

        ex_round = Exposure(2.0, 5; round_offsets=true)
        @test ex_round.round_offsets == true

        # nsteps forced to 1 when exptime is zero
        ex_zero = Exposure(0.0, 10)
        @test ex_zero.nsteps == 1

        # Warning for nsteps=1 with non-zero exptime
        @test_warn "Ignoring non-zero exposure time" Exposure(1.0, 1)
    end

    @testset "CircularAperture" begin
        ap_f32 = CircularAperture(Float32, (16, 16), 7)
        @test eltype(ap_f32) == Float32
        @test all(0f0 .<= ap_f32 .<= 1f0)

        # aa_dist=2: transition band has intermediate values
        ap_aa = CircularAperture((32, 32), 14; aa_dist=2)
        @test any(x -> 0.0 < x < 1.0, ap_aa)
    end

    @testset "num_workers" begin
        @test get_num_threads() == 1
        set_num_threads(nothing)
        @test get_num_threads() == max(Threads.nthreads() ÷ 2, 1)
        set_num_threads(3)
        @test get_num_threads() == 3
        set_num_threads(1) do
            @test get_num_threads() == 1
        end
        set_num_threads(nothing) do
            @test get_num_threads() == max(Threads.nthreads() ÷ 2, 1)
        end
    end

    @testset "TrueSky models" begin
        # Test PointSource
        ps = PointSource()
        @test ps isa PointSource

        # Test DoubleSystem
        ds = DoubleSystem((5, 3), 0.5)
        @test ds.rel_position == (5, 3)
        @test ds.intensity == 0.5

        ds_f32 = DoubleSystem((5, 3), Float32(0.5))
        @test ds_f32.intensity isa Float32

        # Test TrueSkyImage
        test_image = rand(32, 32)
        ts_img = TrueSkyImage(test_image)
        @test size(ts_img.true_sky_fft) == (32, 32)

        ts_img_f32 = TrueSkyImage(convert.(Float32, test_image))
        @test eltype(ts_img_f32.true_sky_fft) == ComplexF32
    end

    @testset "ImagingPlan" begin
        ap = CircularAperture((16, 16))
        img_spec = ImagingSpec(ap, 16, ReadoutSpec(1e6, 100))
        atm = SingleLayer(5.0)

        img_plan = ImagingPlan(atm, img_spec)
        @test img_plan.offsets == [(0, 0)]

        img_spec2 = ImagingSpec(ap, 16, ReadoutSpec(1e6, 100); exposure=Exposure(0.1, 10))
        img_plan2 = ImagingPlan(atm, img_spec2)
        @test img_plan2.offsets == [(0, 0)]

        atm2 = SingleLayer(5.0; harding=(; nsteps=:auto), wind_velocity=(10.0, 5.0))
        img_plan3 = ImagingPlan(atm2, img_spec2)
        @test length(img_plan3.offsets) == 10
        @test img_plan3.offsets[1] == (0, 0)
        @test img_plan3.offsets[2] != (0, 0)

        img_spec3 = ImagingSpec(ap, 16, ReadoutSpec(1e6, 100); exposure=Exposure(0, 10))
        img_plan4 = ImagingPlan(atm2, img_spec3)
        @test img_plan4.offsets == [(0, 0)]
    end
end
