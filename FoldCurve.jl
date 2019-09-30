using PseudoArcLengthContinuation, LinearAlgebra, Plots
using DifferentialEquations, Flux, DiffEqFlux
const Cont = PseudoArcLengthContinuation

Plots.scatter(curve::ContResult; kwargs...) = scatter(curve.branch[1,:],curve.branch[2,:]; kwargs...)
function steady_state( rates, u, θ ; kwargs...)
	"""computes limit curve saddle using parameter continuation methods"""

	# integrate until steady state
	steady_state, _, stable = Cont.newton(
		u -> rates(u,-2.0,θ), u, Cont.NewtonPar(tol = 1e-11) )

	branches, _, _ = Cont.continuation( (u,x) -> rates(u,x,θ),
		steady_state, θ, ContinuationPar(maxSteps=0))

	#################################################################################
	if stable # find fold points on forward/backward branches
		for σ ∈ [-1,1]

			branch, _, _ = Cont.continuation(
				(u,x) -> rates(u,x,θ),

				steady_state, θ, ContinuationPar(ds=σ*0.01; kwargs...),
				printsolution = u -> u[1] )

			append!( branches.branch, branch.branch )
			append!( branches.bifpoint, branch.bifpoint )
		end
	else
		printstyled(color=:red,"[error] integration to steady state failed\n")
		return branches
	end
	return branches
end


function rates(u,θ1,θ2)
	return  Array([ θ2-θ1 + (θ2+θ1)*u[1] - u[1]^3 ])
end

function target(u)
	return u-u^3
end

function cost(curve::ContResult)
	norm( curve.branch[1,:] - map(u -> target(u), curve.branch[2,:] ) )
end

θ = 0.1
u = [0.0]

curve = steady_state(
	(u,a,b) -> rates(u,a,b), u,θ;
	maxSteps=2000, pMin=-2.0, pMax=2.0 )

scatter( curve, label="inferred", color="darkblue",
	markerstrokewidth=0,markersize=3)

scatter!( map(u -> target(u), curve.branch[2,:] ), curve.branch[2,:],
	label="target", color="gold", markersize=3, markerstrokewidth=0,
	xlabel="parameter", ylabel="steady state")


function fit(u,θ1,θ2)

	θ1,θ2 = param(θ1),param(θ2)
	u = param([u])

	function loss()

		curve = steady_state(
			(u,a,b) -> rates(u,a,b), u,θ1,θ2;
			maxSteps=2000, pMin=-2.0, pMax=2.0 )

		return cost(curve)
	end

  iter = Iterators.repeated((), 20)
  # Callback function to observe training
  function cb()

	  scatter( curve, label="inferred", color="darkblue",
	  	markerstrokewidth=0,markersize=3)

	  scatter!( map(u -> target(u), curve.branch[2,:] ), curve.branch[2,:],
	  	label="target", color="gold", markersize=3, markerstrokewidth=0,
	  	xlabel="parameter", ylabel="steady state")
  end

  @time Flux.train!(loss, Flux.Params([θ2]), iter, ADAM(0.2), cb=cb)
end

fit(0.0,-1.0,-1.0)

θ2 = param([θ2])

function loss()

	# curve = steady_state(
	# 	(u,θ1,θ2) -> rates(u,θ1,θ2), u,θ1,θ2;
	# 	maxSteps=2000, pMin=-2.0, pMax=2.0 )

	return cost(curve)+θ2
end

loss()

u = param(0.0)

u^3
