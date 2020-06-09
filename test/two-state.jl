######################################################## model
function rates(u::Vector{T},parameters::NamedTuple{(:θ,:p,:n),Tuple{Vector{U},U,V}}) where {T<:Number,U<:Number,V<:Int}
	@unpack p,θ,n = parameters; y₂,μ₁,μ₂,a₁,a₂,b₁,b₂ = θ
	return [ a₁/(1+(p*u[4])^n) - μ₁*u[1],  b₁*u[1] - μ₂*u[2],
			 a₂/(1+(y₂*u[2])^n) - μ₁*u[3],  b₂*u[3] - μ₂*u[4] ]
end

function rates(u::CuArray{T},p::CuArray{T},parameters::NamedTuple{(:θ,:p,:n),Tuple{Vector{U},U,V}}) where {T<:Number,U<:Number,V<:Int}
	@unpack θ,n = parameters; y₂,μ₁,μ₂,a₁,a₂,b₁,b₂ = θ
	return a₁./(1 .+ (p' .* u[4,:]).^n ) .- μ₁.*u[1,:],  b₁.*u[1,:] .- μ₂.*u[2,:], a₂./(1 .+ (y₂.*u[2,:]).^n ) .- μ₁.*u[3,:],  b₂.*u[3,:] .- μ₂.*u[4,:]
end

function determinant(u::CuArray{T},p::CuArray{T},parameters::NamedTuple{(:θ,:p,:n),Tuple{Vector{U},U,V}}) where {T<:Number,U<:Number,V<:Int}
	@unpack θ,n = parameters; y₂,μ₁,μ₂,a₁,a₂,b₁,b₂ = θ
	return p' .* ( θ₁ .+ 3θ₂.*u[1,:].^2 ) ./ p'
end

function curvature(u::CuArray{T},p::CuArray{T},parameters::NamedTuple{(:θ,:p,:n),Tuple{Vector{U},U,V}}) where {T<:Number,U<:Number,V<:Int}
	@unpack θ,n = parameters; y₂,μ₁,μ₂,a₁,a₂,b₁,b₂ = θ
	return p' .* ( θ₁ .+ 3θ₂.*u[1,:].^2 ) ./ p'
end

######################################################### initialise targets, model and hyperparameters
targetData = StateDensity(0:0.05:7,cu([5.0,4.0]))
hyperparameters = getParameters(targetData)

u₀ = [[4.0 3.0 0.0 0.0], [4.0 3.0 0.0 0.0]]
parameters = ( θ=[2.5, 0.5, 7.5, 4.0, 2.0, 0.4, 1.5], p=minimum(targetData.parameter), n=2)
