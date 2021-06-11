using FiniteDiff: finite_difference_gradient
using BifurcationInference, Test

error_tolerance = 0.05
@testset "Gradients" begin

	@testset "Saddle Node" begin
		include("minimal/saddle-node.jl")

		@time for θ ∈ [[5.0,1.0],[5.0,-1.0],[-1.0,-1.0],[2.5,-1.0],[-1.0,2.5],[-4.0,1.0]]
			x, y = finite_difference_gradient(θ->loss(F,θ,X),θ), ∇loss(F,θ,X)[2]
			@test acos(x'y/(norm(x)*norm(y)))/π < error_tolerance
		end
	end

	@testset "Pitchfork" begin
		include("minimal/pitchfork.jl")

		@time for θ ∈ [[1.0,1.0],[3.0,3.0],[1.0,-1.0],[3.0,-3.0],[-1.0,-1.0],[-3.0,-3.0],[-1.0,1.0],[-3.0,3.0]]
			x, y = finite_difference_gradient(θ->loss(F,θ,X),θ), ∇loss(F,θ,X)[2]
			@test acos(x'y/(norm(x)*norm(y)))/π < error_tolerance
		end
	end

	@testset "Two State" begin
		include("applied/two-state.jl")

		@time for θ ∈ [[0.5, 0.5, 0.5470, 2.0, 7.5],[0.5, 0.5, 1.5470, 2.0, 7.5],[0.5, 0.88, 1.5470, 2.0, 7.5]]
			x, y = finite_difference_gradient(θ->loss(F,θ,X),θ), ∇loss(F,θ,X)[2]
			@test acos(x'y/(norm(x)*norm(y)))/π < error_tolerance
		end
	end
end