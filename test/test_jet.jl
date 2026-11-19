@testset "JET Precompilation Report" begin
    jet_report = JET.report_package(SpeckleSimulator; toplevel_logger=nothing,
        ignored_modules=[HDF5, ChunkSplitters])
    print(jet_report)
    @test length(JET.get_reports(jet_report)) <= 17
    @test_broken length(JET.get_reports(jet_report)) == 0
end
