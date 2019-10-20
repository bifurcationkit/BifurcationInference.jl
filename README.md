# FluxContinuation

The purpose of this library is to extend the methods provided in `PseudoArcLengthContinuation.jl` to be used together with `Flux.jl`. This way the continuation method can be used as a layer in a machine learning proceedure and we can run inference problems directly on the geometry of state space.

## Co-dimension one methods
Extending the one parameter continuation `Cont.continuation` which returns a `ContResult` type and now also contains `TrackedReal` types. This can be called in the same way `Cont.continuation` is called. See `tests/continuation.jl` for an example.
```
continuation( f::Function, J::Function, u₀::Vector{T}, p₀::T, printsolution = u->u[1] ; kwargs...)
```

## Kernel Density Estimation
These methods are extended so that we may essentially have a differenitable histogram method.
```
kde( ::Array{TrackedReal},  ... )
```

## Dependencies
please check out in-development dependency

`KernelDensity.jl` from `https://github.com/gszep/KernelDensity.jl/tree/relax-types`

the standard `kde` from master branch is not compatible with `Flux.jl`
