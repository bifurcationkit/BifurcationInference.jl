using PseudoArcLengthContinuation, Setfield, Parameters, Plots
using Flux: gradient, params
using Zygote: Buffer, @adjoint, @nograd
using Dates: now

@nograd now,string
include("patches.jl")
include("solvers.jl")

######################### define model and continuation method
θ,k=[-1.0],[2.0]
F = (x, p) -> (@. p + x - (θ+x)^(k+1)/(k+1))
J = (x, p) -> 1 .- (θ+x)[:,:].^k

function steady_states() global opts,iter

	opts = ContinuationPar(
		dsmax = 0.1, dsmin = 1e-3, ds = 0.001, pMin = -3., pMax = 1.0, computeEigenValues = true, detectBifurcation = true,
		newtonOptions = NewtonPar(linsolver=LinearSolver(),
		eigsolver=EigenSolver())
	)
	iter = PALCIterable(F, J, [3.0], -3.0, (@lens _), opts, BorderedLinearSolver(); verbosity=0)

	resp = Buffer(Float64[])
	resx = Buffer(Float64[])

	for state in iter
		push!(resx, getx(state)[1])
		push!(resp, getp(state))
	end

	return copy(resp), copy(resx)
end

######################### minimal test
plot(steady_states()...; label = "", xlabel = "p")

dθ = gradient(params(θ)) do
	p,u = steady_states()
	sum(p.^2+u.^2)
end
