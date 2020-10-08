using BifurcationKit: AbstractLinearSolver, AbstractBorderedLinearSolver, AbstractEigenSolver, _axpy

############################################################################ hyperparameter updates
function getParameters(data::StateDensity{T}; maxIter::Int=100, tol=1e-5) where {T<:Number}
    return ContinuationPar{T,LinearSolver,EigenSolver}(

        pMin=minimum(data.parameter),pMax=maximum(data.parameter), maxSteps=10*length(data.parameter),
        ds=step(data.parameter), dsmax=step(data.parameter), dsmin=step(data.parameter),

            newtonOptions = NewtonPar( linsolver=LinearSolver(), eigsolver=EigenSolver(),
            verbose=false,maxIter=maxIter,tol=T(tol)),

        detectFold = false, detectBifurcation = true)
end

function updateParameters!(parameters::ContinuationPar{T, S, E}, steady_states::Vector{Branch{T}};
    resolution=400 ) where {T<:Number, S<:AbstractLinearSolver, E<:AbstractEigenSolver}

    # estimate scale from steady state curves
    branch_points = map(length,steady_states)
    ds = maximum(branch_points)*parameters.ds/resolution
    parameters = setproperties(parameters;ds=ds,dsmin=ds,dsmax=ds)
end

############################################################################# non-mutating solvers for BifurcationKit
struct EigenSolver <: AbstractEigenSolver end
function (l::EigenSolver)(J, nev::Int64)
	F = eigen(Array(J))
	return Complex.(F.values), Complex.(F.vectors), true, 1
end

struct LinearSolver <: AbstractLinearSolver end
function (l::LinearSolver)(J, rhs; a₀ = 0, a₁ = 1, kwargs...)
	return _axpy(J, a₀, a₁) \ rhs, true, 1
end
function (l::LinearSolver)(J, rhs1, rhs2; a₀ = 0, a₁ = 1, kwargs...)
	return J \ rhs1, J \ rhs2, true, (1, 1)
end

@with_kw struct BorderedLinearSolver{S<:AbstractLinearSolver} <: AbstractBorderedLinearSolver
	solver::S = LinearSolver()
end
function (lbs::BorderedLinearSolver{S})( J, dR, dzu, dzp::T, R, n::T,
		xiu::T = T(1), xip::T = T(1); shift::Ts = nothing)  where {T, S, Ts}

	x1, x2, _, (it1, it2) = lbs.solver(J, R, dR)
	dl = (n - dot(dzu, x1) * xiu) / (dzp * xip - dot(dzu, x2) * xiu)
	x1 = x1 .- dl .* x2

	return x1, dl, true, (it1, it2)
end

############################################################################# training loop
function train!( F::Function, u₀::Vector{Array{T,2}}, parameters::NamedTuple, data::StateDensity;
				iter::Int=200, optimiser=Momentum(0.001), plot_solution = false ) where T<:Number

	Loss = steady_states = NaN
	trajectory = typeof(parameters.θ)[]

	hyperparameters = getParameters(data)
	∇Loss = similar(parameters.θ)

	for i=1:iter
		try
			steady_states = deflationContinuation(F,u₀,parameters,(@lens _.p),hyperparameters)
			Loss,∇Loss = ∇loss(Ref(F),steady_states,Ref(parameters.θ),data.bifurcations)

		catch
			printstyled(color=:red,   "Iteration $i\tSkipped\n") end
			printstyled(color=:yellow,"Iteration $i\tLoss = $Loss\n")

		println("Parameters\t$(parameters.θ)")
		println("Gradients\t$(∇Loss)")

		update!(optimiser, parameters.θ, ∇Loss )
		push!(trajectory,copy(parameters.θ))
		if plot_solution if i%plot_solution==0 plot(steady_states,data) end end
	end

	return trajectory
end

############################################################################## loss evaluation helper
function loss(F::Function, θ::AbstractVector{T}, data::StateDensity, u₀::Vector{Vector{Vector{T}}}, hyperparameters::ContinuationPar; λ::T=0.0, ϵ::T=0.1) where T<:Number 
	parameters = (θ=θ,p=minimum(data.parameter))

	try 
		steady_states = deflationContinuation(F,u₀,parameters,(@lens _.p),hyperparameters)
		return loss(Ref(F),steady_states,Ref(θ),data.bifurcations; λ=λ,ϵ=ϵ)

	catch
		return NaN
	end
end

############################################################################# plotting
import Plots: plot
function plot(steady_states::Vector{Branch{T}}, data::StateDensity{T}) where {T<:Number,U<:Number}
	right_axis = plot(steady_states; displayPlot=false)

	vline!( data.bifurcations.x, label="", color=:gold)
	plot!( right_axis,[],[], color=:gold, legend=:bottomleft, alpha=1.0, label="") |> display
end

function plot(steady_states::Vector{Branch{T}}; displayPlot=true) where {T<:Number}

	plot([NaN],[NaN],label="",xlabel=L"\mathrm{parameter,}p", right_margin=20mm,size=(500,400))
	right_axis = twinx()

    for branch in steady_states

        stability = map( λ -> all(real(λ).<0), branch.eigvals)
        determinants = map( λ -> prod(real(λ)), branch.eigvals)

        plot!(right_axis, branch.parameter, determinants, linewidth=2, alpha=0.5, label="", grid=false,
        	ylabel=L"\mathrm{determinant}\,\,\Delta_{\theta}(u,p)",
            color=map( stable -> stable ? :red : :pink, stability )
		)
			
		for idx ∈ 1:length(first(branch.state))

			plot!(branch.parameter, map(x->x[idx],branch.state), linewidth=2, alpha=0.5, label="", grid=false,
				ylabel=L"\mathrm{steady\,states}\quad F_{\theta}(u,p)=0",
				color=map( stable -> stable ? :darkblue : :lightblue, stability )
			)

			scatter!( branch.parameter[branch.bifurcations],
				map(x->x[idx],branch.state)[branch.bifurcations],
				label="", m = (3.0,3.0,:black,stroke(0,:none))
			)
		end
    end

	if displayPlot
		plot!(right_axis,[],[], color=:red, legend=:bottomleft, alpha=1.0, label="", linewidth=2) |> display
	else
		plot!(right_axis,[],[], color=:red, legend=:bottomleft, alpha=1.0, label="", linewidth=2)
		return right_axis
	end
end