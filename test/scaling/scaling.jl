######################################################## model
function rates(u::AbstractVector{T},parameters::NamedTuple,N::Int,M::Int) where T<:Number

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

targetData = StateDensity(-π:0.01:π,Ref([0.0]))
hyperparameters = getParameters(targetData)

function benchmark(N::Int,M::Int)
	F = (u,p) -> rates(u,p,N,M)

	parameters = ( θ=ones(M), p=-π )
	u₀ = [ [ones(N)], [-ones(N)] ]

	steady_states = deflationContinuation( F, u₀, parameters, (@lens _.p), hyperparameters )
	θ = Ref(parameters.θ)
	F = Ref(F)

	@time ∇loss( F, steady_states, θ, targetData.bifurcations )
	return @elapsed ∇loss( F, steady_states, θ, targetData.bifurcations )
end

pyplot()
heatmap( 1:2, 1:10:41, benchmark,
	xlabel="States", ylabel="Parameters", size=(500,400),
	colorbar_title="Iteration Execution / sec"
)

savefig(joinpath(@__DIR__,"scaling.pdf"))
