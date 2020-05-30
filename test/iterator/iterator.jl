using PseudoArcLengthContinuation, Setfield, Parameters, Plots
using Flux: gradient, params

using Zygote: Buffer, @adjoint, @adjoint!, @nograd
using Dates: now

@nograd now,string
include("patches.jl")
include("solvers.jl")

######################### define model and continuation method
function rates(u::Vector,parameters::NamedTuple)
	@unpack p,θ = parameters; r,α,c = θ
	θ₁,θ₂ = r*cos(α), r*sin(α)
	return [ p[1] + θ₁*u[1] + θ₂*u[1]^3 + c ]
end

function rates_jacobian(u::Vector,parameters::NamedTuple)
	@unpack θ = parameters; r,α = θ
	θ₁,θ₂ = r*cos(α), r*sin(α)
	return [ θ₁ + 3 *θ₂*u[1]^2 ][:,:]
end

parameters = ( p=[-3.0,0.0], θ=[3.0,5.0,0.0] )
u = [-5.0]

function states(parameters,u=[-5.0])

	hyperparameters = ContinuationPar( maxSteps=2000, detectBifurcation = true,
		dsmax = 0.01, dsmin = 1e-3, ds = 0.001, pMin = -3.1, pMax = 3.1,
		computeEigenValues = true, saveEigenvectors = false, inPlace = false,

		newtonOptions = NewtonPar( maxIter=1000,
			linsolver=LinearSolver(), eigsolver=EigenSolver()
		)
	)

	linsolver = BorderedLinearSolver()
	iterator = PALCIterable( rates, rates_jacobian, u, parameters, (@lens _.p[1]), hyperparameters, linsolver)

	x = Buffer(Float64[])
	p = Buffer(Float64[])

	for state in iterator
		push!(x, getx(state)[1])
		push!(p, getp(state))
	end

	return copy(p),copy(x)
end

plot( states(parameters)..., label="x", xlabel="p")

gradients, = gradient(parameters) do parameters
	p,x = states(parameters)
	sum(abs.(diff(x))) + sum(abs.(diff(p)))
end

	@assert gradients.θ != nothing
		@assert ~iszero(gradients.θ)
