module Environments
using ReinforcementLearningEnvironments
using ReinforcementLearningBase
using Random

"""
    create_cartpole_env(;kwargs...)

Creates a data structure to hold a Cartpole environment, including a current state.

This is provided by ReinforcementLearningEnvironments.jl
"""
function create_cartpole_env(;kwargs...)
    return CartPoleEnv(;kwargs...)
end


action_space(env::CartPoleEnv) = env.action_space;
state_space(env::CartPoleEnv) = env.observation_space;
reward(env::CartPoleEnv{A,T}) where {A, T} = env.done ? zero(T) : one(T);
is_terminated(env::CartPoleEnv) = env.done;
state(env::CartPoleEnv) = env.state;
reset!(env::CartPoleEnv) = ReinforcementLearningBase.reset!(env);
step!(env::CartPoleEnv, action)= ReinforcementLearningEnvironments._step!(env, action)

export action_space, state_space, reward, is_terminated, state, reset!, step!, create_cartpole_env

end