include("helper.jl")

# Define the name of the policies
get_policy_name(::RandomPolicy) = "Random Policy"

function plot_random_policy_returns(epochs=128)
    policy = RandomPolicy();
    ts = get_cartpole_trajectories(policy, epochs)
    returns = get_total_return.(ts)
    plt = plot(returns, legend=false)
    title!("Random Policy Cartpole Return")
    xlabel!("Epoch")
    ylabel!("Return")
    return plt
end