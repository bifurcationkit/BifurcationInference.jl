using FluxContinuation: continuation
using Flux, Zygote

using PseudoArcLengthContinuation: plotBranch,ContinuationPar,NewtonPar,DefaultLS,DefaultEig
using Plots, LinearAlgebra, Printf

# parametrised hypothesis
function f( u,p, θ₁=0.0, θ₂=0.0, θ₃=0.0)
	return [ p + θ₁*u[1] + θ₂*u[1]^3 + θ₃, -u[2] ]
end

# jacobian
function J( u,p, θ₁=0.0, θ₂=0.0, θ₃=0.0)
	return [[ θ₁ .+ 3 .*θ₂.*u[1].^2, 0.0 ] [ 0.0, -1.0 ]]
end

target(u) = u .- u.^3

################################################## test inference
function infer( f::Function, J::Function, θ::AbstractArray, target )

	# hyperparameters
	parameters = ContinuationPar{Float64, typeof(DefaultLS()), typeof(DefaultEig())}(
		pMin=-2.0,pMax=2.0,ds=0.01, maxSteps=1000,

			newtonOptions = NewtonPar{Float64, typeof(DefaultLS()), typeof(DefaultEig())}(
			verbose=false,maxIter=1000,tol=1e-10),

		computeEigenValues = false)

	u₀ = [2.0,0.0]
	bifurcations,u₀ = continuation( f,J,u₀, parameters )

	function loss()
		bifurcations,u₀ = continuation( f,J,u₀, parameters )
		P,U = bifurcations.branch[1,:], bifurcations.branch[2,:]
		return norm( P .- target(U) )
	end

	function progress()
		@printf("Loss = %f, θ = [%f,%f,%f], u₀ = [%f,%f]\n", loss(), θ..., u₀...)
		P,U = bifurcations.branch[1,:], bifurcations.branch[2,:]

		plotBranch( bifurcations, label="inferred")
		plot!( target(U), U,
			label="target", color="gold",
			xlabel="parameter, p", ylabel="steady state") |> display
	end

	@time Flux.train!( loss, Zygote.Params([θ]), 100, ADAM(0.1); cb=progress)
end

θ = [-2.0,1.0,0.0]
infer( (u,p)->f(u,p,θ...), (u,p) -> J(u,p,θ...), θ, target)
