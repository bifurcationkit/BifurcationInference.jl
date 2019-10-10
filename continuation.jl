using Flux, NLsolve, LinearAlgebra
include("utils.jl")

"""
continuation (u₀,p₀) -> (uMax,pMax) such that f(u,p) = 0
returns array containing columns (u,p,∂ₚu)
"""
function continuation( f, u₀, p₀ ; kwargs...)

	# estimate initial tangent
	u,p, ∂ₚu = tangent( f, u₀, p₀ ; kwargs... )
	ds = kwargs[:ds]; ∂ₛp = 1.0
	∂ₛu = ∂ₚu

	# psuedo-arclength constraint
	constraint = (u₀,p₀,u,p) -> (u-u₀)*∂ₛu + (p-p₀)*∂ₛp - ds
	""" correction to prediction (u,p) from (u₀,p₀) """
	function solve(u₀,p₀,u,p)
		solver = nlsolve( z -> ( f(z...), constraint(u₀,p₀,z...) ), [u,p],
			factor=1.0/norm([u,p]), autoscale=false )

		if solver.f_converged return solver.zero
		else throw("initial_tanget solver : not converged") end
	end

	# main continuation loop
	output = typeof(u)[u,p,∂ₚu]
	while (p <= kwargs[:pMax]) & (abs(u) <= kwargs[:uRange])

		# predictor
		u₊ = u + ∂ₛu * ds
		p₊ = p + ∂ₛp * ds

		# corrector
		u₊,p₊ = solve(u,p,u₊,p₊)

		# update
		∂ₛu = ( u₊ - u ) / ds
		∂ₛp = ( p₊ - p ) / ds

		u,p, ∂ₚu = u₊,p₊, ∂ₛu/∂ₛp
		output = hcat(output,[u,p,∂ₚu])
	end

	if typeof(u) <: Tracker.TrackedReal
		return Tracker.collect(output) # TrackedArray

	else
		return output # Array
	end
end
