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
	function solve(u₀,p)

		solver = nlsolve( z -> ( f(z...), z[2]-p ), [u₀,p],
			factor=1.0/norm([u₀,p]), autoscale=false )
		u,p = solver.zero

		if solver.f_converged return u else

			solver = nlsolve( z -> ( f(z...), z[2]-p ), [-u₀,p],
				factor=1.0/norm([-u₀,p]), autoscale=false )
			u,p = solver.zero

			if solver.f_converged return u
			else throw("initial_tanget solver : not converged") end
		end
	end

	u₀ = solve(u₀,p₀)
	du = solve(u₀,p₀+dp) - u₀
	return u₀,p₀, du/dp
end
