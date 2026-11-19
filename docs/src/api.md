# API Reference

```@meta
CurrentModule = SpeckleSimulator
```

```@index
Modules = [SpeckleSimulator]
```

## Setting up the atmosphere model
```@autodocs
Modules = [SpeckleSimulator]
Pages = ["atmosphere.jl"]
Filter = t -> t !== samplephases!
```

```@docs
samplephases!
```

## Imaging specifications
```@autodocs
Modules = [SpeckleSimulator]
Pages = ["imaging.jl"]
```

## Running simulations
```@autodocs
Modules = [SpeckleSimulator]
Pages = ["simulation.jl", "io.jl"]
```
