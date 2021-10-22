using Plots.PlotMeasures
using LaTeXStrings
using Plots

import Plots: plot,plot!
function plot(steady_states::Vector{<:Branch},data::StateSpace; kwargs...)

	layout = @layout [a;b{1.0w,0.5h}]
	default(); default( ; grid=false,label="",margin=1mm,linewidth=2,lims=:round, kwargs...)

	figure = plot(layout = layout, link = :x, xlim=extrema(data.parameter), size=(300,500), ylabel=L"\mathrm{steady\,\,states}\quad |\,u\,\,|" )
	hline!( [0],subplot=2,linewidth=1,color=:black, xlabel=L"\mathrm{control\,\,condition,}p",
		xmirror=true, topmargin=-5mm, ylabel=L"\mathrm{spectrum}\,\quad \rho(\lambda)")
	
	vline!(data.targets,subplot=1,linewidth=1,color=:gold)
	vline!(data.targets,subplot=2,linewidth=1,color=:gold)

	for branch ∈ steady_states

		alpha = map(s->window_function(data.parameter,s.z),branch)
		parameter = map( s -> s.z.p, branch)
		states = map(s-> norm(s.z.u), branch)

		realExtrema = map( s->extrema(real(s.λ)), branch)
		imagExtrema = map( s->extrema(imag(s.λ)), branch)

		plot!( parameter, states, subplot=1, alpha=alpha, color=:lightgreen,
			linewidth=map(r->any(λ->λ≠0,r) ? 5 : 0, imagExtrema) )

		plot!( parameter, states, subplot=1, alpha=alpha,
			color=map(r->last(r)<0 ? :darkblue : :lightblue, realExtrema) )

		plot!( parameter, map(r->asinh(first(r)),imagExtrema), fillrange=map(r->asinh(last(r)),imagExtrema),
			subplot=2, alpha=alpha, color=:lightgreen, fillalpha=0.25*alpha, linewidth=0 )

		plot!( parameter, map(r->asinh(last(r)),realExtrema), ribbon=( map(r->asinh(last(r))-asinh(first(r)),realExtrema), zeros(length(parameter))),
			color=map(r->last(r)<0 ? :darkblue : :lightblue, realExtrema),
			subplot=2, alpha=alpha, fillalpha=0.2*alpha )
	end

	plot!( [NaN], [NaN], fillrange=[NaN], color=:lightgreen, fillalpha=0.25, linewidth=0, subplot=2, label=L"\Im\mathrm{m}\lambda" )
	plot!( [NaN], [NaN], fillrange=[NaN], color=:darkblue, fillalpha=0.2, subplot=2, label=L"\Re\mathrm{e}\lambda" )
	xticks!([NaN],subplot=1)
	return figure
end

function plot!(steady_states::Vector{<:Branch},data::StateSpace; determinant=false, kwargs...)
 
	hline!([0];linewidth=0,color=:black, ylabel=L"\mathrm{steady\,\,states}\quad F_{\theta}(u,p)=0", xlabel=L"\mathrm{control\,\,condition,}p", kwargs...)
	vline!(data.targets;linewidth=1,color=:gold, kwargs...)
	for branch ∈ steady_states

		stability = map( s -> all(real(s.λ).<0), branch)
		parameter = map( s -> s.z.p, branch)

		for idx ∈ 1:dim(branch)

			plot!( parameter, map(s->s.z.u[idx],branch); grid=false,label="",margin=1mm,linewidth=2,
				color=map( stable -> stable ? :darkblue : :lightblue, stability ), kwargs... )
		end
	end
end

function plot(F::Function, θ::AbstractVector, data::StateSpace; kwargs...)

	parameters = (θ=θ,p=minimum(data.parameter))
	hyperparameters = getParameters(data)

	steady_states = deflationContinuation(F,data.roots,parameters,hyperparameters)
	println(steady_states)

	return plot(steady_states,data;kwargs...)
end

function plot!(F::Function, θ::AbstractVector, data::StateSpace; kwargs...)

	parameters = (θ=θ,p=minimum(data.parameter))
	hyperparameters = getParameters(data)

	steady_states = deflationContinuation(F,data.roots,parameters,hyperparameters)
	println(steady_states)

	return plot!(steady_states,data;kwargs...)
end