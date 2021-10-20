## utility script to generate data for the phase retrieval application
# generated data:
# signal with 30% non-zero with Exp(1.0) distribution
# W has random Gaussian elements on which abs was applied
# signal noise has standard deviation 50 times lower than signal std

import MAT
using Distributions
using SparseArrays
using Random
using Statistics: std

problem_sizes = [(1000, 1000), (2000, 1000), (2000, 5000), (5000, 5000)]
noise_level = (50, 25, 10)

d = Distributions.Exponential(1.0)

Random.seed!(42)
for (n, p) in problem_sizes
    for nlev in noise_level
        θ = sprand(Random.GLOBAL_RNG, n, 0.3, (r, i) ->  rand(r, d, i))
        W = 5.0 * abs.(randn(Random.GLOBAL_RNG, n, p))
        y_noisy = W' * θ
        σ_sig = std(y_noisy) * inv(nlev)
        y_noisy .+= σ_sig * randn(length(y_noisy))
        res = Dict("theta" => θ, "W" => W, "y" => y_noisy)
        MAT.matwrite(joinpath(@__DIR__, "data/phase_retrieval/data_$(n)_$(p)_$(nlev).mat"), res)
    end
end
