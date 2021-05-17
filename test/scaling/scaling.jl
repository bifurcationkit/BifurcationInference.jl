######################################################## model
function F(u::AbstractVector{T},parameters::NamedTuple,N::Int,M::Int) where T<:Number

	@unpack θ,p = parameters
	f = first(u)*first(p)*first(θ)
	F = zeros(typeof(f),length(u))

	F[1] = sin(p)^2 - ( θ[1]*sin(p)^2 + 1 )*u[1]
	for i ∈ 2:N
		F[i] = u[i-1] - u[i]
	end

	for i ∈ 2:M
		F[1+mod(i-1,N)] -= u[1+mod(i-1,N)]*θ[i]
	end

	return F
end

function scaling(N::Int,M::Int)
	F(u,p) = F(u,p,N,M)

	X = StateSpace(N,-π:0.01:π,[0.0])
	θ = SVector{M}(ones(M))

	println("N = $N M = $M")
	@time ∇loss(F,θ,X)
	return @elapsed ∇loss(F,θ,X)
end