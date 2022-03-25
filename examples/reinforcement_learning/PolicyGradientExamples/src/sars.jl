struct SARS{S,A,R}
    state::S
    action::A
    reward::R
    next_state::S
end

get_reward(transition::SARS) = transition.reward;
get_total_return(trajectory) = mapreduce(get_reward, +, trajectory)