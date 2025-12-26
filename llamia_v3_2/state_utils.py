
def get_current_state(state, node_id):
    return state.get(str(node_id), {})
