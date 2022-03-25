using Plots
using PolicyGradientExamples


function get_cartpole_trajectories(policy, n::Integer=1)
    trajectories = []
    env = create_cartpole_env()
    for _ = 1:n
        reset!(env)
        push!(trajectories, run_episode(policy, env))
    end
    return [trajectories...]
end