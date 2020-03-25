include("_inference.jl")

######################################################## model
function rates( u,p, θ₁=0.0, θ₂=0.0, θ₃=0.0, θ₄=0.0, θ₅=0.0, θ₆=0.0, θ₇=0.0, θ₈=0.0)
	return [ θ₁*p - ( θ₂ + θ₃*u[2] )*u[1],
			 θ₄*(1-u[2])/(0.01+1-u[2]) - (θ₆+θ₇*u[1])*u[2]/(0.01+u[2]) ]
end

function jacobian( u,p, θ₁=0.0, θ₂=0.0, θ₃=0.0, θ₄=0.0, θ₅=0.0, θ₆=0.0, θ₇=0.0, θ₈=0.0)
	return [[ θ₂ + θ₃*u[2] , θ₃*u[1] ] [ -θ₇*u[2]/(u[2]+0.01), -0.01*θ₄/(0.01+1-u[2])^2 - 0.01*(θ₆+θ₇*u[1])/(0.01+u[2])^2 ]]
end

# target state density
parameter = -2:0.1:2
density = ones(length(parameter)).*(abs.(parameter).<0.5)

######################################################## run inference
θ = param(randn(3))
f = (u,p)->rates(u,p,θ...)
J = (u,p)->jacobian(u,p,θ...)

infer( f, J, θ, StateDensity(parameter,density); iter=200)
