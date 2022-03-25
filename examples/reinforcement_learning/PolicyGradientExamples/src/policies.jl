module Policies
using ..Environments
using Random
using ReinforcementLearningEnvironments


struct RandomPolicy{T<:AbstractRNG}
    rng::T
end

RandomPolicy() = RandomPolicy(Random.GLOBAL_RNG);

action(policy::RandomPolicy, env) = rand(policy.rng, action_space(env));
probability(::RandomPolicy, env, action) = 1.0/length(action_space(env));
function probability(::RandomPolicy, env) 
    a_space = action_space(env)
    p = 1.0/length(a_space)
    return ones(Float64, size(a_space)) .* p
end

export action, probability, RandomPolicy
end