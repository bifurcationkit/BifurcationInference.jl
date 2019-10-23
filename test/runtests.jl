# throw("unit tests currently fail due to N^2 dft algorithm")
# tests = [ "univariate", "bivariate", "interp" ]
# println("Testing KernelDensity.jl ...")

# for test in tests
#     println(" * $test.jl")
#     include("$test.jl")
# end


tests = ["continuation"]
println("Testing FluxContinuation.jl ...")

for test in tests
    println(" * $test.jl")
    include("$test.jl")
end

# include("../patches/KernelDensity.jl")
# using FFTW: fft,rfft,ifft,irfft
# using Flux

# input,output = 100*rand(101),rand(51)
# N = length(input)

# @assert all( fft(input) .≈ fft(param.(input)) )
# @assert all( ifft(input) .≈ ifft(param.(input)) )

# @assert all(input .≈ ifft(fft(input)))
# @assert all(input .≈ ifft(fft(param.(input))))

# @assert all( rfft(input) .≈ rfft(param.(input)) )
# @assert all( irfft(output,N) .≈ irfft(param.(output),N) )

# @assert all(input .≈ irfft(rfft(input),length(input)))
# @assert all(input .≈ irfft(rfft(param.(input)),length(input)))

# input,output = 100*rand(100),rand(51)
# N = length(input)

# @assert all( fft(input) .≈ fft(param.(input)) )
# @assert all( ifft(input) .≈ ifft(param.(input)) )

# @assert all(input .≈ ifft(fft(input)))
# @assert all(input .≈ ifft(fft(param.(input))))

# @assert all( rfft(input) .≈ rfft(param.(input)) )
# @assert all( irfft(output,N) .≈ irfft(param.(output),N) )

# @assert all(input .≈ irfft(rfft(input),length(input)))
# @assert all(input .≈ irfft(rfft(param.(input)),length(input)))
