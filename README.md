# FluxContinuation.jl

This library implements the method described in **Szep, G. Dalchau, N. and Csikasz-Nagy, A. 2020. "Inference of Bifurcations with Differentiable Continuation" [pre-print work in progress]** using parameter continuation library [`BifurcationKit.jl`](https://github.com/rveltz/BifurcationKit.jl) and auto-differentiation [`Zygote.jl`](https://github.com/FluxML/Zygote.jl). This way continuation methods can be used as layers in machine learning proceedures, and inference can be run on problems directly on the geometry of state space. Watch my [JuliaCon video](https://www.youtube.com/watch?v=vp-206RgeVE) for a description of the work so far :)

[![Build Status](https://travis-ci.com/gszep/FluxContinuation.jl.svg?branch=master)](https://travis-ci.com/gszep/FluxContinuation.jl)
[![Coverage](https://codecov.io/gh/gszep/FluxContinuation.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/gszep/FluxContinuation.jl)
