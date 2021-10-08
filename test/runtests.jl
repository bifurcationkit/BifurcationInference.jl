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

using Flux: Optimise
error_tolerance = 0.1

@testset "Optimisations" begin
include("applied/two-state.jl")

	@testset "Mutual Inhibition" begin
		parameters = ( θ=SizedVector{5}(0.5,0.5,0.5470,1.0,1.5), p=minimum(X.parameter) )
		trajectory = train!( F, parameters, X;  iter=300, optimiser=Optimise.ADAM(0.01) )

		steady_states = deflationContinuation(F,X.roots,(p=minimum(X.parameter),θ=trajectory[end]),getParameters(X))
		bifurcations = unique([ s.z for branch ∈ steady_states for s ∈ branch if s.bif ], atol=3*step(X.parameter) )

		@test all( z->any(isapprox.(X.targets,z.p; atol=error_tolerance)), bifurcations)
	end

	@testset "Mutual Activation" begin
		parameters = ( θ=SizedVector{5}(0.246,-0.113,-0.87,-1.36,1.28), p=minimum(X.parameter) )
		trajectory = train!( F, parameters, X;  iter=300, optimiser=Optimise.ADAM(0.01) )

		steady_states = deflationContinuation(F,X.roots,(p=minimum(X.parameter),θ=trajectory[end]),getParameters(X))
		bifurcations = unique([ s.z for branch ∈ steady_states for s ∈ branch if s.bif ], atol=3*step(X.parameter) )

		@test all( z->any(isapprox.(X.targets,z.p; atol=error_tolerance)), bifurcations)
	end
end

include("applied/double-exclusive.jl")
plot(F,θ,X)

parameters = ( θ=θ, p=minimum(X.parameter) )
trajectory = train!( F, parameters, X;  iter=200, optimiser=Optimise.ADAM(0.001) )

plot(F,trajectory[end],X)