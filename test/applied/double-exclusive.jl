######################################################## model
F(z::BorderedArray,θ::AbstractVector) = F(z.u,(θ=θ,p=z.p))
function F(u::AbstractVector,parameters::NamedTuple)
	@unpack θ,p = parameters
	θ = Dict(

		# morphogens
		"c₆" => 10^p,
		"c₁₂" => 10^θ[1],
	
		# dissociation constants
		"R⁻P76" => 10^θ[2],
		"S⁻P76" => 10^θ[3],
		"R⁻P81" => 10^θ[4],
		"S⁻P81" => 10^θ[5],
	
		# crosstalk
		"S⁻C6" => 10^θ[6],
		"R⁻C12" => 10^θ[7],
	
		# leaky expression
		"ε₇₆" => 10^θ[8],
		"ε₈₁" => 10^θ[9],
	
		# hill coefficients
		"nᴿ" => 10^θ[10],
		"nˢ" => 10^θ[11],
		"nᴸ" => 10^θ[12],
		"nᵀ" => 10^θ[13],
	
		# synthesis rates
		"aᴿ" => 10^θ[14],
		"aˢ" => 10^θ[15],
		"aᴸ" => 10^θ[16],
		"aᵀ" => 10^θ[17],
	
		# degredations
		"dᴿ" => 10^θ[18],
		"dˢ" => 10^θ[19],
		"dᴸ" => 10^θ[20],
		"dᵀ" => 10^θ[21],
	)
 
	return F( 10 .^ u, θ )
end

##############################  receiver binding
bound(x, cₓ, cₑ, ε, nₓ)  = x * x * ( abs(cₓ) ^ nₓ + abs(ε*cₑ) ^ nₓ )
hill(x, n)  = 1 / (1 + abs(x)^n)

##############################  promoter activations
function P76( u::AbstractVector, θ::Dict )
	@unpack R⁻C12,S⁻C6, nˢ,nᴿ, R⁻P76,S⁻P76,ε₇₆, c₆,c₁₂ = θ
	R,S,L,T = u

	activation = R⁻P76 * bound( R, c₆, c₁₂, R⁻C12, nᴿ ) + S⁻P76 * bound( S, c₁₂, c₆, S⁻C6, nˢ )
	return  (ε₇₆ + activation) / (1 + activation)
end

function P81( u::AbstractVector, θ::Dict )
	@unpack R⁻C12,S⁻C6, nˢ,nᴿ, R⁻P81,S⁻P81,ε₈₁, c₆,c₁₂ = θ
	R,S,L,T = u

	activation = R⁻P81 * bound( R, c₆, c₁₂, R⁻C12, nᴿ ) + S⁻P81 * bound( S, c₁₂, c₆, S⁻C6, nˢ )
	return (ε₈₁ + activation) / (1 + activation)
end

function F( u::AbstractVector, θ::Dict )
	@unpack aᴿ,aˢ,aᴸ,aᵀ, nᵀ,nᴸ, dᴿ,dˢ,dᴸ,dᵀ = θ

	f = first(u)*first(values(θ))
	F = similar(u,typeof(f))

	# calculate reaction rates
	F[1] = aᴿ * hill(u[4],nᵀ) - (1+dᴿ) * u[1]
	F[2] = aˢ * hill(u[3],nᴸ) - (1+dˢ) * u[2]

	F[3] = aᴸ * P76(u,θ) - (1+dᴸ) * u[3]
	F[4] = aᵀ * P81(u,θ) - (1+dᵀ) * u[4]

	return F
end

######################################################### targets and initial guess
X = StateSpace( 4, 0:0.01:3, [1,2] )
X.roots .= [ [[-1.2019347487897403, -1.279773518365126, 1.1551118279729176, 0.5901366314351191]],[[-0.3334508216225101, -2.070943259128422, 2.640454683781306, -0.2372608628463465]]]

θ = SizedVector{21}(0.0,

	# dissociation constants
	-1.465,
	-3.782,
	-3.129,
	1.017,

	# crosstalk
	-8.0,
	-7.0,

	# leaky expression
	-1.943,
	-2.573,

	# hill coefficients
	-0.209,
	-0.489,
	-0.236,
	0.217,

	# synthesis rates
	1.222,
	0.456,
	3.394,
	1.828,

	# degredations
	1.390,
	0.933,
	0.0,
	0.0,
)