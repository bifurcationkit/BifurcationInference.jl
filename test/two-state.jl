######################################################## model
function rates(u::Vector,θ::NamedTuple)
	@unpack p,z,n = θ; y₂,μ₁,μ₂,a₁,a₂,b₁,b₂ = z
	return [ a₁/(1+(p*u[4])^n) - μ₁*u[1],  b₁*u[1] - μ₂*u[2],
			 a₂/(1+(y₂*u[2])^n) - μ₁*u[3],  b₂*u[3] - μ₂*u[4] ]
end

function rates_jacobian(u::Vector,θ::NamedTuple)
	@unpack p,z,n = θ; y₂,μ₁,μ₂,a₁,a₂,b₁,b₂ = z

	_2to3 = -n*u[2]^(n-1)*a₂*y₂^n/(1+(y₂*u[2])^n)^2
	_4to1 = -n*u[4]^(n-1)*a₁*p^n/(1+(p*u[4])^n)^2

	return [[ -μ₁,    b₁,  0.0,  0.0  ] [ 0.0,   -μ₂, _2to3, 0.0  ] [ 0.0,   0.0,   -μ₁,  b₂  ] [ _4to1, 0.0,   0.0, -μ₂  ]]
end

function curvature(u::Vector,p::Number;θ::NamedTuple=θ)
	@unpack z,n = θ; y₂,μ₁,μ₂,a₁,a₂,b₁,b₂ = z
	return 0.0
end

######################################################### initialise targets, model and hyperparameters
targetData = StateDensity(0:0.05:7,[5.0,4.0])
hyperparameters = getParameters(targetData)

u₀ = [[4.0 3.0 0.0 0.0], [4.0 3.0 0.0 0.0]]
θ = ( z=[2.5, 0.5, 7.5, 4.0, 2.0, 0.4, 1.5], p=[minimum(targetData.parameter)], n=2)
