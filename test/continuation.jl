using FluxContinuation: continuation,unpack
using Flux, Plots, LinearAlgebra, Printf

# parametrised hypothesis
function f( u,p, θ₁=0.0, θ₂=0.0, θ₃=0.0)
	return [ p + θ₁*u[1] + θ₂*u[1]^3 + θ₃, -u[2] ]
end

# jacobian
function J( u,p, θ₁=0.0, θ₂=0.0, θ₃=0.0)
	return [[ θ₁ .+ 3 .*θ₂.*u[1].^2, 0.0 ] [ 0.0, -1.0 ]]
end

target(u) = u .- u.^3

################################################## unit tests

function loss(f,J,u₀,p₀)
	bifurcations, = continuation( f,J, u₀,p₀;
		pMin=-2.1, pMax=2, ds=0.01,
		maxSteps=1000, maxIter=1000,
		computeEigenValues=true )

	P,U, = unpack(bifurcations)
	return norm( P .- target(U) )
end

# untracked test
θ = [-2.0,1.0,0.0]
u₀,p₀ = [1.5,0.0], -2.0
loss( (u,p) -> f(u,p,θ...), (u,p) -> J(u,p,θ...), u₀,p₀)

# tracked test
θ = param([-2.0,1.0,0.0])
u₀,p₀ = param.([1.5,0.0]), param(-2.0)
loss( (u,p) -> f(u,p,θ...), (u,p) -> J(u,p,θ...), u₀,p₀)


################################################## test inference
function infer( f::Function, J::Function, θ::TrackedArray, target )

	# initialise hyperparameters
	# ALL inputs must be tracked variables
	u₀,p₀ = param.([1.5,0.0]), param(-2.0)
	bifurcations, = continuation( f,J,u₀,p₀;
		pMin=-2.1, pMax=2, ds=0.01,
		maxSteps=1000, maxIter=1000,
		computeEigenValues=true )

	function loss()
		bifurcations, = continuation( f,J,u₀,p₀;
			pMin=-2.1, pMax=2, ds=0.01,
			maxSteps=1000, maxIter=1000,
			computeEigenValues=true )

		P,U, = unpack(bifurcations)
		return norm( P .- target(U) )
	end

	function progress()
		@printf("Loss = %f, θ = [%f,%f,%f], u₀ = [%f,%f]\n", loss(), θ.data...,u₀...)
		P,U, = unpack(bifurcations)

		plotBranch( bifurcations, label="inferred")
		plot!( target(Tracker.data.(U)),Tracker.data.(U),
			label="target", color="gold",
			xlabel="parameter, p", ylabel="steady state") |> display
	end

	@time Flux.train!(loss, Flux.Params([θ]),
		Iterators.repeated((),100), ADAM(0.1);
		cb=progress)
end

θ = param([-2.0,1.0,0.0])
infer( (u,p)->f(u,p,θ...), (u,p) -> J(u,p,θ...), θ, target)