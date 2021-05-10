function finite_differences(θ::AbstractVector{<:Number}; order=5, geom=true, condition=100)
	loss(rates,θ,targetData,hyperparameters)
	return first(grad(central_fdm(order,1,geom=geom,condition=condition),
		θ -> loss(rates,θ,targetData,hyperparameters), θ ))
end

function autodiff(θ::AbstractVector{<:Number})
	L,∇L = ∇loss(rates,θ,targetData,hyperparameters)
	return ∇L
end