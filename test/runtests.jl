using FluxContinuation, StaticArrays
using Flux: Optimise

using FiniteDifferences, Test
using LinearAlgebra: norm
using StatsBase: median

include("utils.jl")

@testset "Minimal Models" begin

	@testset "Saddle Node" begin include("minimal/saddle-node.jl")
		@test @time random_test(rates,targetData)
	end

	@testset "Pitchfork" begin include("minimal/pitchfork.jl")
		@test @time random_test(rates,targetData)
	end
end
