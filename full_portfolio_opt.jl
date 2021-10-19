using MAT
using Plots
using LinearAlgebra
using FrankWolfe
using ReverseDiff
using JSON

# NOTE: the data are random normal matrices with mean 0.05, not 0.1 as indicated in their paper
# we also generated additional datasets at larger scare and log-normal revenues

include("monotonous_step.jl")

portfolio_dir = readdir(joinpath(@__DIR__, "data/portfolio/"), join=true)
# re-adding first dataset to avoid measuring compilation
push!(portfolio_dir, portfolio_dir[1])
raw_data = map(portfolio_dir) do f
    MAT.matread(f)["W"]
end

function build_objective(W)
    (n, p) = size(W)
    function f(x)
        -sum(log(dot(x, @view(W[:,t]))) for t in 1:p)
    end
    function ∇f(storage, x)
        storage .= 0
        for t in 1:p
            temp_rev = dot(x, @view(W[:,t]))
            @. storage -= @view(W[:,t]) ./ temp_rev
        end
        storage
    end
    (f, ∇f)
end


# lower bound on objective value
true_obj_value = -13

for data_idx in eachindex(raw_data)
    W = raw_data[data_idx]
    (f, ∇f) = build_objective(W)
    
    lmo = FrankWolfe.ProbabilitySimplexOracle(1.0)
    x0 = FrankWolfe.compute_extreme_point(lmo, rand(size(W, 1)))
    storage = Vector{Float64}(undef, size(x0)...)
    (x, v, primal_agnostic, dual_gap, traj_data_agnostic) = FrankWolfe.frank_wolfe(
        x -> f(x) - true_obj_value, ∇f, lmo, x0,
        verbose=true,
        trajectory=true,
        line_search=FrankWolfe.Agnostic(),
        max_iteration=5000,
        gradient=storage,
    )

    (xback, v, primal_back, dual_gap, traj_data_backtracking) = FrankWolfe.frank_wolfe(
        x -> f(x) - true_obj_value, ∇f, lmo, x0,
        verbose=true,
        trajectory=true,
        line_search=FrankWolfe.Adaptive(),
        max_iteration=5000,
        gradient=storage,
    )

    (xback, v, primal_back, dual_gap, traj_data_monotonous) = FrankWolfe.frank_wolfe(
        x -> f(x) - true_obj_value, ∇f, lmo, x0,
        verbose=true,
        trajectory=true,
        line_search=MonotonousStepSize(),
        max_iteration=5000,
        gradient=storage,
    )

    (xaw, v, primal_aw, dual_gap, traj_data_aw) = FrankWolfe.away_frank_wolfe(
        x -> f(x) - true_obj_value, ∇f, lmo, x0,
        verbose=true,
        trajectory=true,
        line_search=FrankWolfe.Adaptive(),
        max_iteration=5000,
        gradient=storage,
    )

    open(joinpath(@__DIR__, "results", split(portfolio_dir[data_idx], "/")[end] * ".json"), "w") do f
        write(f,
            JSON.json((
                agnostic=traj_data_agnostic,
                backtracking=traj_data_backtracking,
                awaystep=traj_data_aw,
                monotonous=traj_data_monotonous,
            )),
        )
    end
end

# experimenting with BigFloat
for raw_file in portfolio_dir
    if occursin("syn_1000_1200_10_50_", raw_file)
        W = MAT.matread(raw_file)["W"]
        (f, ∇f) = build_objective(W)

        lmo = FrankWolfe.ProbabilitySimplexOracle(big(1.0))
        x0 = FrankWolfe.compute_extreme_point(lmo, rand(size(W, 1)))
        storage = Vector{BigFloat}(undef, size(x0)...)

        (x, v, primal_agnostic, dual_gap, traj_data_agnostic) = FrankWolfe.frank_wolfe(
            x -> f(x) - true_obj_value, ∇f, lmo, x0,
            verbose=true,
            trajectory=true,
            line_search=FrankWolfe.Agnostic(),
            max_iteration=5000,
            gradient=storage,
        )

        (xback, v, primal_back, dual_gap, traj_data_backtracking) = FrankWolfe.frank_wolfe(
            x -> f(x) - true_obj_value, ∇f, lmo, x0,
            verbose=true,
            trajectory=true,
            line_search=FrankWolfe.Adaptive(),
            max_iteration=5000,
            gradient=storage,
        )

        (xback, v, primal_back, dual_gap, traj_data_monotonous) = FrankWolfe.frank_wolfe(
            x -> f(x) - true_obj_value, ∇f, lmo, x0,
            verbose=true,
            trajectory=true,
            line_search=MonotonousStepSize(),
            max_iteration=5000,
            gradient=storage,
        )

        (xaw, v, primal_aw, dual_gap, traj_data_aw) = FrankWolfe.away_frank_wolfe(
            x -> f(x) - true_obj_value, ∇f, lmo, x0,
            verbose=true,
            trajectory=true,
            line_search=FrankWolfe.Adaptive(),
            max_iteration=5000,
            gradient=storage,
        )

        open(joinpath(@__DIR__, "results", split(raw_file, "/")[end] * "_big.json"), "w") do f
            write(f,
                JSON.json((
                    agnostic=traj_data_agnostic,
                    backtracking=traj_data_backtracking,
                    awaystep=traj_data_aw,
                    monotonous=traj_data_monotonous,
                )),
            )
        end
    end
end
