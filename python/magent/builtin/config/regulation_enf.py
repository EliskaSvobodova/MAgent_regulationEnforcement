from python import magent


def get_config(map_size):
    gw = magent.gridworld
    cfg = gw.Config()

    cfg.set({"map_width": map_size, "map_height": map_size})

    compliant = cfg.register_agent_type(
        "compliant",
        {'speed': 1, 'view_range': gw.CircleRange(2)}
    )
    defective = cfg.register_agent_type(
        "defective",
        {'speed': 1, 'view_range': gw.CircleRange(2)}
    )

    compliant_group = cfg.add_group(compliant)
    defective_group = cfg.add_group(defective)

    return cfg