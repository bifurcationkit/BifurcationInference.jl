################################################################################ likelihood defition
function likelihood( F::Function, z::BorderedArray, θ::AbstractVector, targets::AbstractVector; kwargs...)
	return gaussian_mixture(targets,z; kwargs...) * bifucation_weight(F,z,θ)
end

function gaussian_mixture( targets::AbstractVector, z::BorderedArray; ϵ=0.1, kwargs...)
	errors = ( targets .- z.p ) .^ 2
	return sum(exp.(-errors/ϵ)) / (length(targets)*√(ϵ*π))
end

function bifucation_weight( F::Function, z::BorderedArray, θ::AbstractVector )
	return exp( -det(∂Fu(F,z,θ))^2 )
end

################################################################################# loss function
function loss( F::Ref{<:Function}, branches::AbstractVector{<:Branch}, θ::Ref{<:AbstractVector}, targets::Ref{<:AbstractVector}; kwargs...)
	return -log(sum( marginal_likelihood.(F,branches,θ,targets;kwargs...) )) + log(sum( normalisation.(F,branches,θ) ))
end

function marginal_likelihood( F::Function, branch::Branch, θ::AbstractVector, targets::AbstractVector; kwargs...)
	return branch.ds'likelihood.( Ref(F), branch.solutions, Ref(θ), Ref(targets); kwargs...)
end

################################################################################ normalisations
function normalisation( F::Function, branch::Branch, θ::AbstractVector )
	return branch.ds'bifucation_weight.( Ref(F), branch.solutions, Ref(θ))
end
