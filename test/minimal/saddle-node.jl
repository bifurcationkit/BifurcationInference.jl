######################################################## model
F(z::BorderedArray,θ::AbstractVector) = F(z.u,(θ=θ,p=z.p))
function F(u::AbstractVector,parameters::NamedTuple)
	@unpack θ,p = parameters

	f = first(u)*first(p)*first(θ)
	F = similar(u,typeof(f))

	F[1] = p + θ[1]*u[1] + θ[2]*u[1]^3
	return F
end

######################################################### targets and initial guess
X = StateSpace( 1, -2:0.01:2, [1,-1] )
θ = SizedVector{2}(5.0,-0.93)
