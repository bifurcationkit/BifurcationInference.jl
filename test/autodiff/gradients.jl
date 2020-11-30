using InvertedIndices,LinearAlgebra,ForwardDiff

########################################################## gradients
function ∇integrand( z::AbstractVector, θ::AbstractVector ) # application of the general leibniz rule
	∇integrand = ForwardDiff.gradient(θ->integrand(z,θ),θ) + deformation(z,θ)'ForwardDiff.gradient(z->integrand(z,θ),z)
	return ∇integrand + integrand(z,θ)*∇region(z,θ)
end

################################### gradient terms due to changing integration region dz
function ∇region( z::AbstractVector, θ::AbstractVector )

	∇deformation = reshape( ForwardDiff.jacobian(z->deformation(z,θ),z), length(z),length(θ),length(z) )
	tangent = tangent_field(z,θ)

	∇region = similar(θ)
	for k ∈ 1:length(θ)
		∇region[k] = tangent'∇deformation[:,k,:]tangent
	end

	return ∇region
end

deformation( z::AbstractVector, θ::AbstractVector) = -∂Fz(z,θ)\∂Fθ(z,θ)
function tangent_field( z::AbstractVector, θ::AbstractVector) where T<:Number
	∂F = ∂Fz(z,θ) # construct tangent field T(z) := det[ ẑ , ∂Fz ]
	field = [ (-1)^(zi+1) * det(∂F[:,Not(zi)]) for zi ∈ 1:length(z) ] # cofactor expansion
	return field / norm(field) # unit tangent field
end

# parameter jacobian
function ∂Fθ( z::AbstractVector, θ::AbstractVector )
	return ForwardDiff.jacobian( θ -> F(z,θ), θ )
end

# augmented statespace jacobian
function ∂Fz( z::AbstractVector, θ::AbstractVector )
	return ForwardDiff.jacobian( z -> F(z,θ) , z )
end
