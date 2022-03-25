module PolicyGradientExamples

include("sars.jl")

include("environments.jl")
include("policies.jl")
include("models.jl")

using .Environments
using .Policies
using .DataStructures

function run_episode(policy, env)
    trajectory = []
    while !is_terminated(env)
        current_state = state(env)
        a = action(policy, env)
        step!(env, a)
        r = reward(env)
        next_state = state(env)
        push!(trajectory, SARS(current_state, a, r, next_state))
    end
    return [trajectory...]
end


export action, probability, RandomPolicy, action_space, state_space, reward, is_terminated, state, reset!, step!, create_cartpole_env, run_episode, SARS, get_total_return, NeuralNetworkDiscretePolicy, get_log_probability_gradients

end