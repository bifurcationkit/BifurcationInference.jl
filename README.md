# BifurcationFit.jl

This library implements the method described in **Szep, G. Dalchau, N. and Csikasz-Nagy, A. 2021. "Parameter Inference with Bifurcation Diagrams" using parameter continuation library [`BifurcationKit.jl`](https://github.com/rveltz/BifurcationKit.jl) and auto-differentiation [`ForwardDiff.jl`](https://github.com/JuliaDiff/ForwardDiff.jl). This way continuation methods can be used as layers in machine learning proceedures, and inference can be run end-to-end directly on the geometry of state space.

[![Build Status](https://travis-ci.com/gszep/FluxContinuation.jl.svg?branch=master)](https://travis-ci.com/gszep/FluxContinuation.jl)
[![Coverage](https://codecov.io/gh/gszep/FluxContinuation.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/gszep/FluxContinuation.jl)
