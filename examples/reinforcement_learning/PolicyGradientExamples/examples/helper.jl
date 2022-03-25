using Plots
using StatsPlots
using PolicyGradientExamples
using Statistics

"""
    get_cartpole_trajectories(policy, n::Integer=1)

Get a vector of trajectories using the supplied policy, on the cartpole problem.
"""
function get_cartpole_trajectories(policy, n::Integer=1)
    trajectories = []
    env = create_cartpole_env()
    for _ = 1:n
        reset!(env)
        push!(trajectories, run_episode(policy, env))
    end
    return [trajectories...]
end

"""
    evaluate_policy(policy, n_samples=1024)

Evaluates the supplied policy on the Cartpole problem, with a supplied number of trajectories.
"""
evaluate_policy(policy, n_samples=1024) = mean(get_total_return.(get_cartpole_trajectories(policy, n_samples)))


"""
    compare_policies(policies...;)

Compares some given policies against one another, showing the distribution of their total returns over a number of samples.
"""
function compare_policies(policies...; n_samples=1024, use_box=false, kwargs...)
    plt = plot()
    plt_fn = use_box ? boxplot! : violin!
    for (i, p) in enumerate(policies)
        plt_fn(repeat([i], n_samples), get_total_return.(get_cartpole_trajectories(policy, n_samples)), label=get_policy_name(p))
    end
    return plot(plt; xticks=false, legendposition=:outerright, kwargs...)
end