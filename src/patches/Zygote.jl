import Base: copyto!

############################## allow copyto! for initialisations
@adjoint! function copyto!(x, y)
	x_ = copy(x)
	copyto!(x, y), function (Δ)
		x_ = copyto!(x_, x)
		return (nothing,Δ)
	end
end
