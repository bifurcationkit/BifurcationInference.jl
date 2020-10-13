############################################################################ hyperparameter updates
function getParameters(data::StateDensity{T}; maxIter::Int=800, tol=1e-6) where T<:Number
	newtonOptions = NewtonPar(verbose=false,maxIter=maxIter,tol=T(tol))

	# support for StaticArrays github.com/JuliaArrays/StaticArrays.jl/issues/73
	newtonOptions = @set newtonOptions.linsolver.useFactorization = false

	return ContinuationPar(
        pMin=minimum(data.parameter), pMax=maximum(data.parameter), maxSteps=10*length(data.parameter),
        ds=step(data.parameter), dsmax=step(data.parameter), dsmin=step(data.parameter),
		newtonOptions=newtonOptions, detectFold=false, detectBifurcation=true, saveEigenvectors=false)
end

function updateParameters!(parameters::ContinuationPar{T, S, E}, steady_states::Vector{Branch{V,T}};
    resolution=400 ) where {T<:Number, V<:AbstractVector{T}, S<:AbstractLinearSolver, E<:AbstractEigenSolver}

    # estimate scale from steady state curves
    branch_points = map(length,steady_states)
    ds = maximum(branch_points)*parameters.ds/resolution
    parameters = setproperties(parameters;ds=ds,dsmin=ds,dsmax=ds)
end

############################################################################# training loop
function train!( F::Function, roots::AbstractVector{<:AbstractVector{<:AbstractVector}},
	             parameters::NamedTuple, data::StateDensity;

				 iter::Int=200, optimiser=Momentum(0.001), plot_solution = false,
				 ϵ::Number=0.1, λ::Number=0.0 )

	Loss = steady_states = nothing
	trajectory = typeof(parameters.θ)[]

	hyperparameters = getParameters(data)
	∇Loss = similar(parameters.θ)

	for i=1:iter
		try
			steady_states = deflationContinuation(F,roots,parameters,hyperparameters)
			Loss,∇Loss = ∇loss(Ref(F),steady_states,Ref(parameters.θ),data.bifurcations;ϵ=ϵ,λ=λ)

		catch error
			printstyled(color=:red,   "Iteration $i\tError = $error\n") end
			printstyled(color=:yellow,"Iteration $i\tLoss = $Loss\n")

		printstyled(color=:blue,"$steady_states\n")
		println("Parameters\t$(parameters.θ)")
		println("Gradients\t$(∇Loss)")
		if isinf(Loss) throw("infinite loss; consider increasing ϵ or λ") end

		update!(optimiser, parameters.θ, ∇Loss )
		push!(trajectory,copy(parameters.θ))
		if plot_solution>0 if i%plot_solution==0 plot(steady_states,data) end end
	end

	return trajectory
end

############################################################################## loss evaluation helper
function loss(F::Function, θ::AbstractVector, data::StateDensity,
	          roots::AbstractVector{<:AbstractVector{<:AbstractVector}},
			  hyperparameters::ContinuationPar; λ::Number=0.0, ϵ::Number=0.1)

	try
		parameters = (θ=θ,p=minimum(data.parameter))
		steady_states = deflationContinuation(F,roots,parameters,hyperparameters)
		return loss(Ref(F),steady_states,Ref(θ),data.bifurcations; λ=λ,ϵ=ϵ)

	catch error
		printstyled(color=:red,"$error\n")
		return NaN
	end
end

############################################################################# plotting
import Plots: plot
function plot(steady_states::Vector{Branch{V,T}}, data::StateDensity{T}) where {T<:Number,V<:AbstractVector{T}}
	right_axis = plot(steady_states; displayPlot=false)

	vline!( data.bifurcations.x, label="", color=:gold)
	plot!( right_axis,[],[], color=:gold, legend=:bottomleft, alpha=1.0, label="") |> display
end

function plot(steady_states::Vector{Branch{V,T}}; displayPlot=true) where {T<:Number,V<:AbstractVector{T}}

	plot([NaN],[NaN],label="",xlabel=L"\mathrm{parameter,}p", right_margin=20mm,size=(500,400))
	right_axis = twinx()

    for branch ∈ steady_states

        stability = map( λ -> all(real(λ).<0), branch.eigvals)
        determinants = map( λ -> prod(real(λ)), branch.eigvals)
		parameter = map(z->z.p,branch.solutions)

		for idx ∈ 1:dim(branch)

			plot!( parameter, map(z->z.u[idx],branch.solutions), linewidth=2, alpha=0.5, label="", grid=false,
				ylabel=L"\mathrm{steady\,states}\quad F_{\theta}(u,p)=0",
				color=map( stable -> stable ? :darkblue : :lightblue, stability )
			)
		end

		plot!(right_axis, parameter, determinants, linewidth=2, alpha=0.5, label="", grid=false,
        	ylabel=L"\mathrm{determinant}\,\,\Delta_{\theta}(u,p)",
            color=map( stable -> stable ? :red : :pink, stability )
		)
    end

	if displayPlot
		plot!(right_axis,[],[], color=:red, legend=:bottomleft, alpha=1.0, label="", linewidth=2) |> display
	else
		plot!(right_axis,[],[], color=:red, legend=:bottomleft, alpha=1.0, label="", linewidth=2)
		return right_axis
	end
end

function plot(F::Function, θ::AbstractVector, data::StateDensity,
			  roots::AbstractVector{<:AbstractVector{<:AbstractVector}},
			  hyperparameters::ContinuationPar, save::String="")

	parameters = (θ=θ,p=minimum(data.parameter))
	steady_states = deflationContinuation(F,roots,parameters,hyperparameters)

	plot(steady_states,data)
	if length(save)>0 savefig(joinpath(@__DIR__,save)) end
end
