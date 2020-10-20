################################################################################ likelihood defition
function likelihood( F::Function, z::BorderedArray, θ::AbstractVector, targets::AbstractVector; kwargs...)
	return gaussian_mixture(targets,z; kwargs...) * bifucation_weight(F,z,θ)
end

function gaussian_mixture( targets::AbstractVector, z::BorderedArray; ϵ=0.1, kwargs...)
	return sum( p->exp(-(p-z.p)^2/ϵ), targets ) / (length(targets)*√(ϵ*π))
end

function bifucation_weight( F::Function, z::BorderedArray, θ::AbstractVector )
	return exp( -det(∂Fu(F,z,θ))^2 )
end

################################################################################# loss function
function loss( F::Function, branches::AbstractVector{<:Branch}, θ::AbstractVector, targets::AbstractVector; kwargs...)
	if sum( branch->sum(branch.bifurcations), branches ) ≥ 2length(targets)

		marginals,norms = sum( branch->marginal_likelihood(F,branch,θ,targets;kwargs...), branches ), sum( branch->normalisation(F,branch,θ), branches )
		return log(norms)-log(marginals)
	else

		curvatures = sum( branch->curvature(F,branch,θ), branches )
		return -log(curvatures)
	end
end

function marginal_likelihood( F::Function, branch::Branch, θ::AbstractVector, targets::AbstractVector; kwargs...)
	return branch.ds'likelihood.( Ref(F), branch.solutions, Ref(θ), Ref(targets); kwargs...)
end

################################################################################ normalisations
function normalisation( F::Function, branch::Branch, θ::AbstractVector )
	return branch.ds'bifucation_weight.( Ref(F), branch.solutions, Ref(θ))
end

########################################################################### determinant curvature
function curvature(F::Function,z::BorderedArray,θ::AbstractVector)

	∂det = ForwardDiff.gradient(z->det(∂Fu(F,z,θ)),z)
	∂∂det = ForwardDiff.hessian(z->det(∂Fu(F,z,θ)),z)

	# tangent field basis
	Jz = ForwardDiff.jacobian(z->tangent_field(F,z,θ),z)
	Tz = tangent_field(F,z,θ)

	return (Tz'∂∂det*Tz + ∂det'Jz*Tz)^2
end

function curvature(F::Function,branch::Branch,θ::AbstractVector)
	return branch.ds'curvature.( Ref(F), branch.solutions, Ref(θ) )
end

function tangent_field( F::Function, z::BorderedArray, θ::AbstractVector)

	∂F = ∂Fz(F,z,θ) # construct tangent field T(z) := det[ ẑ , ∂Fz ]
	field = [ (-1)^(zi+1) * det(∂F[:,Not(zi)]) for zi ∈ 1:length(z) ] # cofactor expansion
	return field / norm(field) # unit tangent field
end
