from python import magent


def get_config(map_size):
    gw = magent.gridworld
    cfg = gw.Config()

    cfg.set({"map_width": map_size, "map_height": map_size})

    compliant = cfg.register_agent_type(
        "compliant",
        {'width': 1, 'length': 1, 'speed': 1, 'view_range': gw.CircleRange(3),
         'attack_range': gw.CircleRange(1), 'step_reward': -0.01}
    )
    defective = cfg.register_agent_type(
        "defective",
        {'width': 1, 'length': 1, 'speed': 1, 'view_range': gw.CircleRange(3),
         'attack_range': gw.CircleRange(1), 'step_reward': -0.01}
    )
    apple = cfg.register_agent_type(
        "apple",
        {'width': 1, 'length': 1, 'speed': 0, 'view_range': gw.CircleRange(2),
         'kill_reward': 5, 'attack_range': gw.CircleRange(0)}
    )  # view range cannot be <= 1 -> convolution dimension error

    g_c = cfg.add_group(compliant)
    g_d = cfg.add_group(defective)
    g_a = cfg.add_group(apple)

    c = gw.AgentSymbol(g_c, index='any')
    d = gw.AgentSymbol(g_d, index='any')
    a = gw.AgentSymbol(g_a, index='any')

    cfg.add_reward_rule(gw.Event(c, 'attack', a), receiver=c, value=1)
    cfg.add_reward_rule(gw.Event(d, 'attack', a), receiver=d, value=1)

    return cfg