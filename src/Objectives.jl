################################################################################
function loss( F::Function, branches::AbstractVector{<:Branch}, θ::AbstractVector, targets::StateSpace; kwargs...)

	predictions = unique([ s.z for branch ∈ branches for s ∈ branch if s.bif ], atol=3*step(targets.parameter) )
	λ = length(targets.targets)-length(predictions)

	if λ≠0 
		Φ = measure(F,branches,θ,targets)
		return errors(predictions,targets) - λ*log(Φ)
	else
		return errors(predictions,targets)
	end
end

################################################################################
function errors( predictions::AbstractVector{<:BorderedArray}, targets::StateSpace)
	return mean( p′-> mean( z->(z.p-p′)^2, predictions; type=:geometric ), targets.targets; type=:arithmetic )
end

function measure( F::Function, z::BorderedArray, θ::AbstractVector, targets::StateSpace )
	return window_function(targets.parameter,z) / ( 1 + abs( derivative( z->log(abs(det(∂Fu(F,z,θ)))), z, tangent_field(F,z,θ) ) )^(-1) ) # todo@(gszep) determinant calculation is computational bottleneck
end

################################################################################
function measure( F::Function, branch::Branch, θ::AbstractVector, targets::StateSpace )
	return sum( s -> measure(F,s.z,θ,targets)*s.ds, branch )
end

function measure( F::Function, branches::AbstractVector{<:Branch}, θ::AbstractVector, targets::StateSpace )
	return sum( branch -> measure(F,branch,θ,targets), branches )
end

########################################################## bifurcation distance and velocity
distance(F::Function,z::BorderedArray,θ::AbstractVector) = [ F(z,θ); det(∂Fu(F,z,θ)) ]
function velocity( F::Function, z::BorderedArray, θ::AbstractVector; newtonOptions=NewtonPar(verbose=false,maxIter=800,tol=1e-6) )
	∂implicit, _, _ = newtonOptions.linsolver( -jacobian( z->distance(F,z,θ), z )', [zero(z.u);one(z.p)] )
	return gradient( θ->distance(F,z,θ)'∂implicit, θ )
end

################################################################################
function tangent_field( F::Function, z::BorderedArray, θ::AbstractVector )
	field = kernel(∂Fz(F,z,θ);nullity=length(z.p))
	return norm(field)^(-1) * BorderedArray( field[Not(end)], field[end] ) # unit tangent field
end

################################################################################
using SpecialFunctions: erf
function window_function( parameter::AbstractVector, z::BorderedArray; β::Real=10 )
	pMin,pMax = extrema(parameter)
	return ( 1 + erf((β*(z.p-pMin)-3)/√2) )/2 * ( 1 - ( 1 + erf((β*(z.p-pMax)+3)/√2) )/2 )
end