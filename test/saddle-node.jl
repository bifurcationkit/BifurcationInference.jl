######################################################## model
function rates(u::Vector{T},parameters::NamedTuple{(:θ,:p),Tuple{Vector{U},U}}) where {T<:Number,U<:Number}
	@unpack θ,p = parameters; r,α,c = θ
	θ₁,θ₂ = r*cos(α), r*sin(α)
	return [ p + θ₁*u[1] + θ₂*u[1]^3 + c ]
end

function rates(u::CuArray{T},p::CuArray{T},parameters::NamedTuple{(:θ,:p),Tuple{Vector{U},U}}) where {T<:Number,U<:Number}
	@unpack θ = parameters; r,α,c = θ
	θ₁,θ₂ = r*cos(α), r*sin(α)
	return p .+ θ₁.*u[1,:] .+ θ₂.*u[1,:].^3 .+ c, 0.0
end

function determinant(u::CuArray{T},p::CuArray{T},parameters::NamedTuple{(:θ,:p),Tuple{Vector{U},U}}) where {T<:Number,U<:Number}
	@unpack θ = parameters; r,α,c = θ
	θ₁,θ₂ = r*cos(α), r*sin(α)
	return θ₁ .+ 3θ₂.*u[1,:].^2
end

function curvature(u::CuArray{T},p::CuArray{T},parameters::NamedTuple{(:θ,:p),Tuple{Vector{U},U}}) where {T<:Number,U<:Number}
	@unpack θ = parameters; r,α = θ
	θ₁,θ₂ = r*cos(α), r*sin(α)
	return -6θ₂ .* ( 1 .+ θ₁^2 .- 9θ₂^2 .*u[1,:].^4 ) ./ (1 .+ (θ₁ .+ 3θ₂.*u[1,:].^2).^2 ).^2
end

######################################################### initialise targets, model and hyperparameters
targetData = StateDensity(-2:0.01:2,cu([1.0,-1.0]))
hyperparameters = getParameters(targetData)

u₀ = [[0.0][:,:], [0.0][:,:]]
parameters = ( θ=[5.0,5.3,0.0], p=minimum(targetData.parameter))
