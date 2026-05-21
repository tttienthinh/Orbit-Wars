test = {"step": 0, "num_agents": None}
steps = 0


def agent(obs, test=test):
    global steps
    print(f"Agent called steps: {steps}")
    steps += 1
    # s = global_board["step"]
    # player = obs.player if hasattr(obs, "player") else obs["player"]
    # print(f"Step {s}, Player {player}")
    # global_board["step"] += 1
    return []
