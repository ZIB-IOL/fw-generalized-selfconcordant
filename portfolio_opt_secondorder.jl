using MAT
using Plots
using LinearAlgebra
using FrankWolfe
using ReverseDiff
using JSON

include(joinpath(@__DIR__, "second_order_rule.jl"))
include(joinpath(@__DIR__, "monotonous_step.jl"))
include(joinpath(@__DIR__, "lloo.jl"))

portfolio_dir = readdir(joinpath(@__DIR__, "data/portfolio/"), join=true)
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

function build_hessian(W)
    (n, p) = size(W)
    function hessian(storage, x)
        storage .= 0
        for t in 1:p
            temp_rev = dot(x, @view(W[:,t]))
            for j in 1:n
                for i in 1:n
                    storage[i,j] += (W[i,t] * W[j,t]) / temp_rev^2
                end
            end
        end
    end
end

W = raw_data[1]
hessian = build_hessian(W)
(f, ∇f) = build_objective(W)
x = rand(size(W, 1))

hess_storage = Matrix{Float64}(undef, size(W, 1), size(W, 1))
line_search_second = SecondOrderSelfConcordant(
    3.0, # nu,
    hessian,
    hess_storage,
    2.0, # M
)

lmo = FrankWolfe.ProbabilitySimplexOracle(1.0)
x0 = FrankWolfe.compute_extreme_point(lmo, rand(size(x)...))

true_obj_value = -12.385135127290818

#Run the LLOO algorithm.
sigma_estimator = compute_convexity_parameter(W)
(x, v, primal_so, dual_gap, traj_data_lloo) = frank_wolfe_lloo(
    x -> f(x) - true_obj_value,
    ∇f, 
    lmo,
    x0,
    hessian,
    hess_storage,
    sigma_estimator,
    2.0,
    max_iteration=1,
)

# precompiling with a run of one iteration
FrankWolfe.frank_wolfe(
    x -> f(x) - true_obj_value, ∇f, lmo, x0,
    verbose=false,
    trajectory=false,
    line_search=line_search_second,
    max_iteration=1,
)

FrankWolfe.frank_wolfe(
    x -> f(x) - true_obj_value, ∇f, lmo, x0,
    verbose=false,
    trajectory=false,
    line_search=FrankWolfe.Agnostic(),
    max_iteration=1,
)

FrankWolfe.frank_wolfe(
    x -> f(x) - true_obj_value, ∇f, lmo, x0,
    verbose=false,
    trajectory=false,
    line_search=MonotonousStepSize(),
    max_iteration=1,
)

FrankWolfe.frank_wolfe(
    x -> f(x) - true_obj_value, ∇f, lmo, x0,
    verbose=false,
    trajectory=false,
    line_search=FrankWolfe.Adaptive(),
    max_iteration=1,
)

FrankWolfe.away_frank_wolfe(
    x -> f(x) - true_obj_value, ∇f, lmo, x0,
    verbose=false,
    trajectory=false,
    line_search=FrankWolfe.Adaptive(),
    max_iteration=1,
)


#Run the LLOO algorithm.
sigma_estimator = compute_convexity_parameter(W)
(x, v, primal_so, dual_gap, traj_data_lloo) = frank_wolfe_lloo(
    x -> f(x) - true_obj_value,
    ∇f, 
    lmo,
    x0,
    hessian,
    hess_storage,
    sigma_estimator,
    2.0,
    max_iteration=1000,
)


# warning: extremely slow
(x, v, primal_so, dual_gap, traj_data_so) = FrankWolfe.frank_wolfe(
    x -> f(x) - true_obj_value, ∇f, lmo, x0,
    verbose=true,
    trajectory=true,
    line_search=line_search_second,
    max_iteration=1000,
)

(x, v, primal_agnostic, dual_gap, traj_data_agnostic) = FrankWolfe.frank_wolfe(
    x -> f(x) - true_obj_value, ∇f, lmo, x0,
    verbose=true,
    trajectory=true,
    line_search=FrankWolfe.Agnostic(),
    max_iteration=1000,
)

(x, v, primal_monotonous, dual_gap, traj_data_monotonous) = FrankWolfe.frank_wolfe(
    x -> f(x) - true_obj_value, ∇f, lmo, x0,
    verbose=true,
    trajectory=true,
    line_search=MonotonousStepSize(),
    max_iteration=1000,
)

(x, v, primal_agnostic, dual_gap, traj_data_backtrack) = FrankWolfe.frank_wolfe(
    x -> f(x) - true_obj_value, ∇f, lmo, x0,
    verbose=true,
    trajectory=true,
    line_search=FrankWolfe.Adaptive(),
    max_iteration=1000,
)

(x, v, primal_agnostic, dual_gap, traj_data_aw) = FrankWolfe.away_frank_wolfe(
    x -> f(x) - true_obj_value, ∇f, lmo, x0,
    verbose=true,
    trajectory=true,
    line_search=FrankWolfe.Adaptive(),
    max_iteration=1000,
)


open(joinpath(@__DIR__, "results", split(portfolio_dir[1], "/")[end] * "_second_order_lloo.json"), "w") do f
    write(f, JSON.json((
        backtrack=traj_data_backtrack,
        agnostic=traj_data_agnostic,
        away_step=traj_data_aw,
        second_order=traj_data_so,
        monotonous=traj_data_monotonous,
        lloo=traj_data_lloo,
    )))
end


json_result_dict = JSON.parsefile(joinpath(@__DIR__, "results", split(portfolio_dir[1], "/")[end] * "_second_order.json"))

traj_data_backtrack = json_result_dict["backtrack"]
traj_data_so = json_result_dict["second_order"]
traj_data_agnostic = json_result_dict["agnostic"]
traj_data_aw = json_result_dict["away_step"]
traj_data_monotonous = json_result_dict["monotonous"]
traj_data_lloo = json_result_dict["lloo"]


FrankWolfe.plot_trajectories([traj_data_backtrack, traj_data_agnostic, traj_data_so, traj_data_aw, traj_data_monotonous, traj_data_lloo], ["Backtracking", "Open loop", "Second order", "AW", "Monotonous", "LLOO"], filename="second_order_standard.pdf", xscalelog=true)

## FrankWolfe.plot_trajectories([traj_data_backtrack, traj_data_agnostic, traj_data_aw, traj_data_monotonous10, traj_data_monotonous100, traj_data_monotonous500], ["Backtracking", "Open loop", "AW", "M10", "M100", "M500"], xscalelog=true)