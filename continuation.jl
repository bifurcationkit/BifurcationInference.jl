using Flux, NLsolve, LinearAlgebra
include("utils.jl")

"""
continuation (u₀,p₀) -> (uMax,pMax) such that f(u,p) = 0
returns arrays containing (u,p,∂ₚu)
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
		else throw("corrector solver : not converged, try decreasing ds") end
	end

	# main continuation loop
	U,P = typeof(u)[u],typeof(u)[p]; dp = 0.0
	while (p <= kwargs[:pMax]) & (abs(u) <= kwargs[:uRange])

		# predictor
		u₊ = u + ∂ₛu * ds
		p₊ = p + ∂ₛp * ds

		# corrector
		u₊,p₊ = solve(u,p,u₊,p₊)

		# update
		∂ₛu = ( u₊ - u ) / ds
		∂ₛp = ( p₊ - p ) / ds

		dp += abs(p₊ - p)
		u,p = u₊,p₊

		if dp > ds # record at equal dp intervals
			push!(U,u); push!(P,p)
			dp = 0.0
		end
	end

	if typeof(u) <: Tracker.TrackedReal
		return Tracker.collect(U),Tracker.collect(P) # TrackedArrays

	else
		return U,P # Arrays
	end
end
