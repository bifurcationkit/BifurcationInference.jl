using Plots.PlotMeasures
using LaTeXStrings
using Plots

import Plots: plot
function plot(steady_states::Vector{<:Branch}; padding=0.2, displayPlot=true)

	pMin,pMax = extrema(vcat([ map(s->s.z.p,branch) for branch ∈ steady_states ]...))
	pMin,pMax = ( pMin+(pMin+pMax)*padding )/(1+2padding),( pMax+(pMin+pMax)*padding )/(1+2padding)
	unpadded_parameter = pMin:pMax

	plot(xlabel=L"\mathrm{parameter,}p", grid=false, right_margin=20mm, size=(500,450) )
	right_axis = twinx()

    for branch ∈ steady_states

        stability = map( s -> all(real(s.λ).<0), branch)
        determinants = map( s -> prod(real(s.λ)), branch)
		parameter = map( s -> s.z.p, branch)

		for idx ∈ 1:dim(branch)

			plot!( parameter, map(s->s.z.u[idx],branch), linewidth=2, label="",
				ylabel=L"\mathrm{steady\,states}\quad F_{\theta}(u,p)=0", grid=false,
				alpha=map( s->window_function(unpadded_parameter,s.z), branch ),
				color=map( stable -> stable ? :darkblue : :lightblue, stability )
			)
		end

		plot!(right_axis, parameter, -determinants, linewidth=2, label="", grid=false,
        	ylabel=L"\mathrm{determinant}\,\quad\left|\!\!\!\!\frac{\partial F_{\theta}}{\partial u}\right|",
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
