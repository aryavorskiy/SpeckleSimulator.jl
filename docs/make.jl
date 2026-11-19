using Documenter
using AtmosphericTurbulenceSimulator
using CairoMakie

ENV["GKSwstype"] = "100"

DocMeta.setdocmeta!(AtmosphericTurbulenceSimulator, :DocTestSetup, :(using AtmosphericTurbulenceSimulator); recursive=true)

makedocs(
    sitename = "AtmosphericTurbulenceSimulator.jl",
    modules = [AtmosphericTurbulenceSimulator, Base.get_extension(AtmosphericTurbulenceSimulator, :MakieExt)],
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", "false") == "true",
        assets = [asset("assets/favicon-96x96.png", class=:ico, islocal=true)],
    ),
    pages = [
        "Overview" => "index.md",
        "Examples" => "examples.md",
        "API Reference" => "api.md",
    ],
)

deploydocs(
    repo = "github.com/aryavorskiy/AtmosphericTurbulenceSimulator.jl.git",
    devbranch = "master",
)
