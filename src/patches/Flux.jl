using Zygote: gradient,Params
using Flux.Optimise: update!

import Flux.Optimise: train!
function train!(loss::Function, ps::Params, iter::Int, opt; cb = () -> ())
	for _ in Iterators.repeated((),iter)

		gs = gradient(ps) do
			loss() end

		for p in ps
			update!(opt, p, [gs[p]...]) end

		cb()
	end
end
