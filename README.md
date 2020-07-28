# FluxContinuation

This library implements the method described in Szep, G. Dalchau, N. and Csikasz-Nagy, A. 2020. "Inference of Bifurcations with Differentiable Continuation" using parameter continuation library [`BifurcationKit.jl`](https://github.com/rveltz/BifurcationKit.jl) and auto-differentiation [`Zygote.jl`](https://github.com/FluxML/Zygote.jl). This way continuation methods can be used as layers in machine learning proceedures, and inference can be run on problems directly on the geometry of state space.

## Co-dimension one methods
Extending the one parameter continuation `Cont.continuation` which returns a `ContResult` type and now also contains `TrackedReal` types. This can be called in the same way `Cont.continuation` is called. See `tests/continuation.jl` for an example.
```
continuation( f::Function, J::Function, u₀::Vector{T}, p₀::T, printsolution = u->u[1] ; kwargs...)
```
