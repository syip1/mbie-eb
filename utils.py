def state_to_ind(state, single):
    if single:
        return (state,)
    else:
        return tuple(map(int, state))
