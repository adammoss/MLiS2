using PolicyGradientExamples
using Flux
using ProgressBars
include("helper.jl")

"""
Gets a tuple of SARS transitions and the associated returns
"""
function batched_monte_carlo(policy;batch_size=64, kwargs...) 
    trajectories = get_cartpole_trajectories(policy, batch_size)
    returns = [calculate_returns(t) for t in trajectories]
    return vcat(trajectories...), vcat(returns...)
end

"""
Uses the policy on the Cartpole environment to generate a sample of trajectories, which are used to update the policy via Monte Carlo Reinforce.

Use batch_size as a keyword argument.
"""
function monte_carlo_epoch!(policy::NeuralNetworkDiscretePolicy, optimiser; kwargs...)
    env = create_cartpole_env()
    transitions, returns = batched_monte_carlo(policy; kwargs...)
    grads = get_model_gradient(env, policy, transitions, returns)
    for p in params(policy.model)
       Flux.update!(optimiser, p, -grads[p]) 
    end
    nothing
end



function run_mc_reinforce_experiment(;show_progress=false)
    epochs = 200_000
    n_samples = 256
    batch_size = 64
    policy = create_cartpole_model()
    optimiser = ADAM(0.005)
    returns = []
    iter = show_progress ? ProgressBar(1:epochs) : (1:epochs)
    push!(returns, evaluate_policy(policy, n_samples))
    for e in iter
        monte_carlo_epoch!(policy, optimiser; batch_size)
        push!(returns, evaluate_policy(policy, n_samples))
    end

    return policy, returns
end