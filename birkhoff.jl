using MAT
using Plots
using LinearAlgebra
using FrankWolfe
using ReverseDiff
using JSON
using Random

include("monotonous_step.jl")

portfolio_dir = readdir(joinpath(@__DIR__, "data/portfolio/"), join=true)
push!(portfolio_dir, portfolio_dir[1])

raw_data = map(portfolio_dir) do f
    MAT.matread(f)["W"]
end

USE_BIG = haskey(ENV, "BIG")
@info "extended precision: $USE_BIG"

function build_objective(W, x0, mu)
    (n, p) = size(W)
    w_1 = 100 * @view(W[:,1])
    function f(X)
        dot(x0, X, w_1) + mu/2 * norm((X - I) * x0)^2
    end
    function grad!(storage, X)
        storage .= x0 * w_1'
        storage .+= mu * (X - I) * (x0 * x0')
        storage
    end
    (f, grad!)
end

# sanity check for gradients
# T = rand(2,2)
# W = rand(2,3)
# x0 = rand(2)
# (f, grad!) = build_objective(W, x0, 2.1)
# f(T)
# storage = similar(T)
# grad!(storage, T)
# grad_ref = ReverseDiff.gradient(f, T)

lmo = FrankWolfe.BirkhoffPolytopeLMO()

for data_idx in eachindex(raw_data)
    Random.seed!(42)
    W0 = raw_data[data_idx]
    n0 = size(W0, 1)
    if USE_BIG
        W = big.(W0[1:(n0÷4), :])
        n = size(W, 1)
        x0 = rand(BigFloat, n)
        x0 ./= sum(x0)
        (f, ∇f) = build_objective(W, x0, 1/sqrt(big(n)))
        X0 = FrankWolfe.compute_extreme_point(lmo, rand(BigFloat, n, n))
        storage = Matrix{BigFloat}(undef, n, n)
    else
        W = W0[1:(n0÷4), :]
        n = size(W, 1)
        x0 = rand(n)
        x0 ./= sum(x0)
        (f, ∇f) = build_objective(W, x0, 1/sqrt(n))
        X0 = FrankWolfe.compute_extreme_point(lmo, rand(n,n))
        storage = Matrix{Float64}(undef, n, n)
    end

    (x, v, primal_agnostic, dual_gap, traj_data_agnostic) = FrankWolfe.frank_wolfe(
        f, ∇f, lmo, copy(X0),
        verbose=true,
        trajectory=true,
        line_search=FrankWolfe.Agnostic(),
        max_iteration=1000,
        gradient=storage,
        epsilon=1e-7,
    )

    (x, v, primal_agnostic, dual_gap, traj_data_backtrack) = FrankWolfe.frank_wolfe(
        f, ∇f, lmo, copy(X0),
        verbose=true,
        trajectory=true,
        line_search=FrankWolfe.Adaptive(),
        max_iteration=1000,
        gradient=storage,
        epsilon=1e-7,
    )

    (xback, v, primal_back, dual_gap, traj_data_monotonous) = FrankWolfe.frank_wolfe(
        f, ∇f, lmo, copy(X0),
        verbose=true,
        trajectory=true,
        line_search=MonotonousStepSize(),
        max_iteration=1000,
        gradient=storage,
        epsilon=1e-7,
    )

    (xaw, v, primal_aw, dual_gap, traj_data_aw) = FrankWolfe.away_frank_wolfe(
        f, ∇f, lmo, copy(X0),
        verbose=true,
        trajectory=true,
        line_search=FrankWolfe.Adaptive(),
        max_iteration=1000,
        gradient=storage,
        epsilon=1e-7,
    )
    write_file = if USE_BIG
        joinpath(@__DIR__, "results", split(portfolio_dir[data_idx], "/")[end] * "_birkhoff_big.json")
    else
        joinpath(@__DIR__, "results", split(portfolio_dir[data_idx], "/")[end] * "_birkhoff.json")
    end
    open(write_file, "w") do f
        write(f,
            JSON.json((
                agnostic=traj_data_agnostic,
                backtracking=traj_data_backtrack,
                awaystep=traj_data_aw,
                monotonous=traj_data_monotonous,
            )),
        )
    end
end
