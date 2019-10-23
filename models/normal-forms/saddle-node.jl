
function rates( u,p, θ₁=0.0, θ₂=0.0, θ₃=0.0)
	return [ p + θ₁*u[1] + θ₂*u[1]^3 + θ₃ ]
end

function jacobian( u,p, θ₁=0.0, θ₂=0.0, θ₃=0.0)
	return [[ θ₁ .+ 3 .*θ₂.*u[1].^2 ]]
end