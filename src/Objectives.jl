################################################################################
function loss( F::Function, branches::AbstractVector{<:Branch}, θ::AbstractVector, targets::StateSpace; kwargs...)

	predictions = unique([ s.z for branch ∈ branches for s ∈ branch if s.bif ], atol=3*step(targets.parameter) )
	λ = length(targets.targets)-length(predictions)

	if λ≠0 
		Φ = measure(F,branches,θ)
		return errors(predictions,targets) - λ*log(Φ)
	else
		return errors(predictions,targets)
	end
end

################################################################################
function errors( predictions::AbstractVector{<:BorderedArray}, targets::StateSpace)
	return mean( p′-> mean( z->(z.p-p′)^2, predictions; type=:geometric ), targets.targets; type=:arithmetic )
end

function measure( F::Function, z::BorderedArray, θ::AbstractVector )
	return 1 / ( 1 + abs( derivative( z->log(abs(det(∂Fu(F,z,θ)))), z, tangent_field(F,z,θ) ) )^(-1) ) # todo@(gszep) determinant calculation is computational bottleneck
end

################################################################################
function measure( F::Function, branch::Branch, θ::AbstractVector )
	return sum( s -> measure(F,s.z,θ)*s.ds, branch )
end

function measure( F::Function, branches::AbstractVector{<:Branch}, θ::AbstractVector )
	return sum( branch -> measure(F,branch,θ), branches )
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
function window_function( parameter::AbstractVector, z::BorderedArray; β::Real=1e16, kwargs... )
	pMin,pMax = extrema(parameter)
	return σ(z.p-pMin;β=β) * ( 1 - σ(z.p-pMax;β=β) )
end

function boundaries( parameter::AbstractVector, z::BorderedArray; β::Real=1e16, kwargs... )
	pMin,pMax = extrema(parameter)
	return ℕ(z.p-pMin;β=β) - ℕ(z.p-pMax;β=β)
end

using SpecialFunctions: erf
function σ(x;β=10)
	return ( 1 + erf(β*x/√2) )/2
end

function ℕ(x;β=10)
	return β*exp(-(β*x)^2/2)/√(2π)
end