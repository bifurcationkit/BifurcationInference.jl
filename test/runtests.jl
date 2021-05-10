using FluxContinuation, StaticArrays
using FiniteDifferences, Test
using LinearAlgebra: norm

include("utils.jl")

error_tolerance = 0.06
@testset "Minimal Models" begin

	@testset "Saddle Node" begin
		include("minimal/saddle-node.jl")

		@time for θ ∈ [[5.0,1.0],[5.0,-1.0],[-1.0,-1.0],[2.5,-1.0],[-1.0,2.5],[-4.0,1.0]]
			x, y = autodiff(θ), finite_differences(θ)
			@test x/norm(x) ≈ y/norm(y) rtol = error_tolerance
		end
	end

	@testset "Pitchfork" begin
		include("minimal/pitchfork.jl")

		@time for θ ∈ [[1.0,1.0],[3.0,3.0],[1.0,-1.0],[3.0,-3.0],[-1.0,-1.0],[-3.0,-3.0],[-1.0,1.0],[-3.0,3.0]]
			x, y = autodiff(θ), finite_differences(θ)
			@test x/norm(x) ≈ y/norm(y) rtol = error_tolerance
		end
	end
end