using Documenter
using SpeckleSimulator
using CairoMakie

ENV["GKSwstype"] = "100"

DocMeta.setdocmeta!(SpeckleSimulator, :DocTestSetup, :(using SpeckleSimulator); recursive=true)

makedocs(
    sitename = "SpeckleSimulator.jl",
    modules = [SpeckleSimulator, Base.get_extension(SpeckleSimulator, :MakieExt)],
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
    repo = "github.com/aryavorskiy/SpeckleSimulator.jl.git",
    devbranch = "master",
)
