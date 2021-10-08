using Parameters,StaticArrays,LinearAlgebra
using BifurcationKit: NewtonPar, newton
using ForwardDiff: jacobian

######################################################## model
#F(z::BorderedArray,θ::AbstractVector) = F(z.u,(θ=θ,p=z.p))

F(u::AbstractVector) = F(u,parameters)
function F(u::AbstractVector,parameters::NamedTuple)
	@unpack θ,p = parameters

	f = first(u)*first(p)*first(θ)
	F = similar(u,typeof(f))

	θ₁,θ₂ = θ[1:N*M+N], θ[N*M+N+1:end]
	W₁,b₁ = reshape( θ₁[1:N*M], (N,M)), θ₁[N*M+1:end]
	W₂,b₂ = reshape( θ₂[1:N*M], (M,N)), θ₂[N*M+1:end]

	F .= W₁*asinh.( W₂*u + b₂ ) + b₁
	F[1] += p

	return F
end

######################################################### targets and initial guess
N,M = 10,10
# X = StateSpace( N, 0:0.01:3, [1,2] )
# X.roots .= [ [randn(N)],[randn(N)]]
# θ = SizedVector{2*N*M+N+M}(randn(2*N*M+N+M))

J(u,p) = jacobian(x->F(x,p),u)
function findroot(i::Integer)
	callback(x, f, J, res, it, itlinear, options; k...) = (println("|f| = $(norm(f))	|J| = $(isnothing(J) ? nothing : det(J))");true)
	try 
		_,_,convergence,_ = newton( F, J, randn(N),
			(θ=SizedVector{2*N*M+N+M}(randn(2*N*M+N+M)),p=randn()), NewtonPar(),
			# callback = callback
		)
		convergence ? printstyled(color=:green,"Convergence\n") : printstyled("Failed\n")
		return convergence
	catch
		printstyled(color=:red,"Error\n")
		return false
	end
end

K = 1000
sum(findroot,1:K)/K
