using NLsolve, LinearAlgebra

"""
estimate tangent ∂ₚu at (u₀,p₀) such that f(u,p) = 0 with trust region method
"""
function tangent( f, u₀, p₀ ; kwargs...)

	# (u,p) must be typed like f(u,p)
	u₀,p₀ = typeof(f(u₀, p₀))(u₀),typeof(f(u₀, p₀))(p₀)
	dp = kwargs[:ds]

	"""
	find point u such that f(u,p) = 0 from initial guess u₀ with trust region method
	"""
	function solve(u₀,p; maxTry=1000, searchstd=1.0)

		u = NaN
		for _=1:maxTry

			solver = nlsolve( z -> ( f(z...), z[2]-p ), [u₀,p],
				factor=1.0/norm([u₀,p]), autoscale=false )

			if solver.f_converged
				u,p = solver.zero
				break

			else u₀ += searchstd*randn() end
		end

		if isnan(u) throw("initial_tanget solver : not converged")
		else return u end
	end

	u₀ = solve(u₀,p₀)
	du = solve(u₀,p₀+dp) - u₀
	return u₀,p₀, du/dp
end
