module DataStructures

struct SARS{S,A,R}
    state::S
    action::A
    reward::R
    next_state::S
end

get_state(transition::SARS) = transition.state
get_action(transition::SARS) = transition.action
get_next_state(transition::SARS) = transition.next_state
get_reward(transition::SARS) = transition.reward;
get_total_return(trajectory) = mapreduce(get_reward, +, trajectory)

export SARS, get_state, get_action, get_next_state, get_reward, get_total_return
end