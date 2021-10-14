using Plots.PlotMeasures
using LaTeXStrings
using Plots

import Plots: plot,plot!
function plot(steady_states::Vector{<:Branch},data::StateSpace; determinant=true, kwargs...)

	if determinant

		layout = @layout [a;b{1.0w,0.5h}]
		default(); default( ; grid=false,label="",margin=1mm,linewidth=2, markerstrokecolor=:auto, markersize=1,kwargs...)
		figure = plot(layout = layout, link = :x, size=(300,500) )

		hline!([0],subplot=1,linewidth=0,color=:black, ylabel=L"\mathrm{steady\,\,states}\quad F_{\theta}(u,p)=0")
		hline!([0],subplot=2,linewidth=1,color=:black, xlabel=L"\mathrm{control\,\,condition,}p", xmirror=true, topmargin=-5mm,
			ylabel=L"\mathrm{spectrum}\,\quad \rho{\theta}(\lambda)")
		
		vline!(data.targets,subplot=1,linewidth=1,color=:gold)
		vline!(data.targets,subplot=2,linewidth=1,color=:gold)

		for branch ∈ steady_states

			maxreal = map( s -> maximum(real(s.λ)), branch)
			minreal = map( s -> minimum(real(s.λ)), branch)

			maximag = map( s -> maximum(imag(s.λ)), branch)
			minimag = map( s -> minimum(imag(s.λ)), branch)

			stability = map( λ -> λ<0, maxreal)
			oscillations = map( s -> any(imag(s.λ).>0), branch)

			parameter = map( s -> s.z.p, branch)
			alpha = map(s->window_function(data.parameter,s.z),branch)

			for idx ∈ 1:dim(branch)
				states = map(s->s.z.u[idx], branch)

				mask = @. stability & oscillations
				scatter!( parameter[mask], states[mask], subplot=1, alpha=alpha[mask], color=:lightgray )

				mask = @. stability & ~oscillations
				scatter!( parameter[mask], states[mask], subplot=1, alpha=alpha[mask], color=:darkgray )

				mask = @. ~stability & ~oscillations
				scatter!( parameter[mask], states[mask], subplot=1, alpha=alpha[mask], color=:darkred )

				mask = @. ~stability & oscillations
				scatter!( parameter[mask], states[mask], subplot=1, alpha=alpha[mask], color=:pink )
			end

			plot!( parameter, asinh.(minimag), fillrange=asinh.(maximag), subplot=2, alpha=alpha,
				color=:darkgreen, fillalpha=0.1, linewidth=0 )

			plot!( parameter, asinh.(maxreal), subplot=2, alpha=alpha, fillalpha=0.1,
				color=map(x -> x ? :gray : :red, stability), ribbon=( asinh.(maxreal).-asinh.(minreal), zeros(length(parameter))) )
		end

		xticks!([NaN],subplot=1)
		return figure

	else 
		default(); default(; grid=false,label="",margin=1mm,linewidth=2,kwargs...)
		figure = plot( size=(300,300) )

		hline!([0],linewidth=0,color=:black, ylabel=L"\mathrm{steady\,\,states}\quad F_{\theta}(u,p)=0", xlabel=L"\mathrm{control\,\,condition,}p")
		vline!(data.targets,linewidth=1,color=:gold)

		for branch ∈ steady_states

			stability = map( s -> all(real(s.λ).<0), branch)
			parameter = map( s -> s.z.p, branch)

			for idx ∈ 1:dim(branch)

				plot!( parameter, map(s->s.z.u[idx],branch),
					color=map( stable -> stable ? :darkblue : :lightblue, stability ) )
			end
		end

		return figure
	end
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