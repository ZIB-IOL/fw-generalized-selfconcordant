# generation of additional datasets for portfolio optimization
# First with more dimensions instead of fixed to n=1000, then with log-distributed revenues

using Distributions
using MAT
using Random


Random.seed!(33)

dlog = LogNormal(0.0, 0.5)
for (n, p) in [(1000, 1500), (5000, 2000)]
    W = Matrix{Float64}(undef, n, p)
    rand!(Random.GLOBAL_RNG, dlog, W)
    MAT.matwrite(joinpath(@__DIR__, "data/portfolio/synlog_$(n)_$(p).mat"), Dict("W" => W))
end


d = Normal(1.0, 0.05)

for (n, p) in [(1500, 1500), (5000, 2000), (5000, 5000)]
    W = Matrix{Float64}(undef, n, p)
    rand!(Random.GLOBAL_RNG, d, W)
    MAT.matwrite(joinpath(@__DIR__, "data/portfolio/syn_$(n)_$(p).mat"), Dict("W" => W))
end

