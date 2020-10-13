using FluxContinuation
using Flux: Momentum
using Test

using StaticArrays
using StatsBase: median
using LinearAlgebra

using Plots.PlotMeasures
using LaTeXStrings
using Plots

gr()
include("minimal/minimal.jl")


pyplot()
include("minimal/saddle-node.jl"); hyperparameters = getParameters(targetData)
heatmap( range(eps(Float64)-π,π-eps(Float64),length=100), range(eps(Float64),7,length=10),

    (α,r) -> asinh(loss(rates, SizedVector{3}(r,α,0.0), targetData, u₀, hyperparameters)),
    aspect_ratio=:equal, ylim=(eps(Float64),7), xlim=(-π,π), proj=:polar, legend=false
)

include("minimal/pitchfork.jl"); hyperparameters = getParameters(targetData)
heatmap( range(eps(Float64)-π,π-eps(Float64),length=100), range(eps(Float64),7,length=10),

    (α,r) -> asinh(loss(rates, SizedVector{2}(r,α), targetData, u₀, hyperparameters)),
    aspect_ratio=:equal, ylim=(eps(Float64),7), xlim=(-π,π), proj=:polar, legend=false
)
