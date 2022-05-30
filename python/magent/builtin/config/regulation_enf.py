from python import magent


def get_config(map_size):
    gw = magent.gridworld
    cfg = gw.Config()

    cfg.set({"map_width": map_size, "map_height": map_size})
    cfg.set({"embedding_size": 19})
    cfg.set({"minimap_mode": True})

    apple = cfg.register_agent_type(
        "apple",
        {'width': 1, 'length': 1, 'speed': 0, 'view_range': gw.CircleRange(2),
         'kill_reward': 10, 'hp': 5, 'attack_range': gw.CircleRange(0), 'step_recover': 1}
    )  # view range cannot be <= 1 -> convolution dimension error

    compliant = cfg.register_agent_type(
        "compliant",
        {'width': 1, 'length': 1, 'speed': 1, 'view_range': gw.CircleRange(3),
         'attack_range': gw.CircleRange(1), 'step_reward': -0.01,
         'hp': 300, 'step_recover': 0, 'dead_penalty': -1, 'attack_penalty': -0.1,
         'damage': 3}
    )
    defective = cfg.register_agent_type(
        "defective",
        {'width': 1, 'length': 1, 'speed': 1, 'view_range': gw.CircleRange(5),
         'attack_range': gw.CircleRange(1), 'step_reward': -0.01,
         'hp': 300, 'step_recover': 0, 'dead_penalty': -1, 'attack_penalty': -0.1,
         'damage': 5}
    )

    g_a = cfg.add_group(apple)
    g_c = cfg.add_group(compliant)
    g_d = cfg.add_group(defective)

    c = gw.AgentSymbol(g_c, index='any')
    d = gw.AgentSymbol(g_d, index='any')
    a = gw.AgentSymbol(g_a, index='any')

    # reward for collecting apples
    cfg.add_reward_rule(gw.Event(c, 'attack', a), receiver=c, value=1)
    cfg.add_reward_rule(gw.Event(d, 'attack', a), receiver=d, value=5)

    # negative reward for attacking another agent
    cfg.add_reward_rule(gw.Event(c, 'attack', d), receiver=c, value=-1)
    cfg.add_reward_rule(gw.Event(d, 'attack', c), receiver=d, value=-1)

    return cfg