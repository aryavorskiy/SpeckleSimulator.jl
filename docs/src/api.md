# API Reference

```@meta
CurrentModule = AtmosphericTurbulenceSimulator
```

```@index
Modules = [AtmosphericTurbulenceSimulator]
```

## Setting up the atmosphere model
```@autodocs
Modules = [AtmosphericTurbulenceSimulator]
Pages = ["atmosphere.jl"]
Filter = t -> t !== samplephases!
```

```@docs
samplephases!
```

## Imaging specifications
```@autodocs
Modules = [AtmosphericTurbulenceSimulator]
Pages = ["imaging.jl"]
```

## Running simulations
```@autodocs
Modules = [AtmosphericTurbulenceSimulator]
Pages = ["simulation.jl", "io.jl"]
```
