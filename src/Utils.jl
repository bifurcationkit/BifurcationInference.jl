############################################################################ hyperparameter updates
function getParameters(data::StateSpace{N,T}; maxIter::Int=800, tol=1e-6, padding=0.2, kwargs...) where {N,T<:Number}
	newtonOptions = NewtonPar(verbose=false,maxIter=maxIter,tol=T(tol))

	# support for StaticArrays github.com/JuliaArrays/StaticArrays.jl/issues/73
	newtonOptions = @set newtonOptions.linsolver.useFactorization = false
	padding = padding*(maximum(data.parameter)-minimum(data.parameter))
	return ContinuationPar(

        pMin=minimum(data.parameter)-padding, pMax=maximum(data.parameter)+padding, maxSteps=10*length(data.parameter),
        ds=step(data.parameter), dsmax=step(data.parameter), dsmin=step(data.parameter),

		newtonOptions=newtonOptions, detectFold=false, detectBifurcation=true,
		saveEigenvectors=false, nev=N )
end

function updateParameters!(parameters::ContinuationPar{T, S, E}, steady_states::Vector{Branch{V,T}};
    resolution=400 ) where {T<:Number, V<:AbstractVector{T}, S<:AbstractLinearSolver, E<:AbstractEigenSolver}

    # estimate scale from steady state curves
    branch_points = map(length,steady_states)
    ds = maximum(branch_points)*parameters.ds/resolution
    parameters = setproperties(parameters;ds=ds,dsmin=ds,dsmax=ds)
end

############################################################################# training loop
function train!( F::Function, parameters::NamedTuple, data::StateSpace;
				 iter::Int=200, optimiser=Momentum(0.001), plot_solution = false, kwargs...)

	hyperparameters = getParameters(data;kwargs...)
	Loss = steady_states = nothing

	∇Loss = similar(parameters.θ)
	trajectory = typeof(parameters.θ)[]

	for i=1:iter
		try
			steady_states = deflationContinuation(F,data.roots,parameters,hyperparameters;kwargs...)
			Loss,∇Loss = ∇loss(F,steady_states,parameters.θ,data;kwargs...)

		catch error
			printstyled(color=:red,   "Iteration $i\tError = $error\n") end
			printstyled(color=:yellow,"Iteration $i\tLoss = $Loss\n")

		printstyled(color=:blue,"$steady_states\n")
		println("Parameters\t$(parameters.θ)")
		println("Gradients\t$(∇Loss)")
		if isinf(Loss) throw("infinite loss") end

		update!(optimiser, parameters.θ, ∇Loss )
		push!(trajectory,copy(parameters.θ))
		if plot_solution>0 if i%plot_solution==0 plot(steady_states,data) end end
	end

	return trajectory
end

############################################################################## loss evaluation helper
function loss(F::Function, θ::AbstractVector, data::StateSpace, hyperparameters::ContinuationPar; kwargs...)

	try
		parameters = (θ=θ,p=minimum(data.parameter))
		steady_states = deflationContinuation(F,data.roots,parameters,hyperparameters;kwargs...)
		return loss(F,steady_states,θ,data;kwargs...)

	catch error
		printstyled(color=:red,"$error\n")
		return NaN
	end
end

function loss(F::Function, θ::AbstractVector, data::StateSpace; kwargs...)
	return loss(F,θ,data,getParameters(data); kwargs...)
end

function ∇loss(F::Function, θ::AbstractVector, data::StateSpace, hyperparameters::ContinuationPar; kwargs...)
	parameters = (θ=θ,p=minimum(data.parameter))

	steady_states = deflationContinuation(F,data.roots,parameters,hyperparameters;kwargs...)
	return ∇loss(F,steady_states,θ,data;kwargs...)
end

function ∇loss(F::Function, θ::AbstractVector, data::StateSpace; kwargs...)
	return ∇loss(F,θ,data,getParameters(data); kwargs...)
end

############################################################################# plotting
import Plots: plot
function plot(steady_states::Vector{<:Branch}; padding=0.1, displayPlot=true)

	pMin,pMax = extrema(vcat([ map(s->s.z.p,branch) for branch ∈ steady_states ]...))
	pMin,pMax = ( pMin+(pMin+pMax)*padding )/(1+2padding),( pMax+(pMin+pMax)*padding )/(1+2padding)
	unpadded_parameter = pMin:pMax

	plot(xlabel=L"\mathrm{parameter,}p", grid=false, right_margin=20mm, size=(500,400) )
	right_axis = twinx()

    for branch ∈ steady_states

        stability = map( s -> all(real(s.λ).<0), branch)
        determinants = map( s -> prod(real(s.λ)), branch)
		parameter = map( s -> s.z.p, branch)

		for idx ∈ 1:dim(branch)

			plot!( parameter, map(s->s.z.u[idx],branch), linewidth=2, label="",
				ylabel=L"\mathrm{steady\,states}\quad F_{\theta}(u,p)=0",
				alpha=map( s->window_function(unpadded_parameter,s.z), branch ),
				color=map( stable -> stable ? :darkblue : :lightblue, stability )
			)
		end

		plot!(right_axis, parameter, determinants, linewidth=2, label="",
        	ylabel=L"\mathrm{determinant}\,\,\Delta_{\theta}(u,p)",
			alpha=map( s->window_function(unpadded_parameter,s.z), branch ),
            color=map( stable -> stable ? :red : :pink, stability )
		)
    end

	if displayPlot plot!() |> display
	else return right_axis end
end

function plot(steady_states::Vector{<:Branch}, data::StateSpace)
	right_axis = plot(steady_states; displayPlot=false)
	vline!( data.targets, label="", color=:gold)
	plot!( right_axis,[],[], color=:gold, legend=:bottomleft, alpha=1.0, label="") |> display
end

function plot(F::Function, θ::AbstractVector, data::StateSpace; kwargs...)

	parameters = (θ=θ,p=minimum(data.parameter))
	hyperparameters = getParameters(data;kwargs...)

	steady_states = deflationContinuation(F,data.roots,parameters,hyperparameters;kwargs...)
	println(steady_states)
	plot(steady_states,data)
end
