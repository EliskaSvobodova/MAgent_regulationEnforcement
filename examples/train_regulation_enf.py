import argparse
import collections
import random
import time
import logging as log
from functools import partial
from collections import deque, defaultdict

import numpy as np

import magent
from magent.builtin.tf_model import DeepQNetwork


def play_a_round(env, map_size, food_handle, player_handles, models, print_every, train=True, render=False, eps=None):
    env.reset()

    # add 4 compliant and 1 defective agents on random places on the map
    env.add_agents(food_handle, method="random", n=3)  # apples
    env.add_agents(player_handles[0], method="random", n=4)  # compliant
    env.add_agents(player_handles[1], method="random", n=1)  # defective

    step_ct = 0
    done = False

    n = len(player_handles)

    hist_len = 5
    reward_history = defaultdict(partial(deque, maxlen=hist_len))
    obs = [[] for _ in range(n)]
    ids = [[] for _ in range(n)]
    acts = [[] for _ in range(n)]
    nums = [env.get_num(handle) for handle in player_handles]
    total_reward = [0 for _ in range(n)]
    cur_pos = [None for _ in range(n)]

    print("===== sample =====")
    print("eps %s number %s" % (eps, nums))
    start_time = time.time()
    while not done:
        # take actions for every model
        for i in range(n):
            obs[i] = env.get_observation(player_handles[i])
            ids[i] = env.get_agent_id(player_handles[i])
            cur_pos[i] = env.get_pos(player_handles[i])
            # let models infer action in parallel (non-blocking)
            models[i].infer_action(obs[i], ids[i], 'e_greedy', eps, block=False)
        for i in range(n):
            acts[i] = models[i].fetch_action()  # fetch actions (blocking)
            env.set_action(player_handles[i], acts[i])

        # simulate one step
        done = env.step()

        # sample - boycotting reward shaping
        step_reward = []
        # defective
        i = 1
        rewards = env.get_reward(player_handles[i])
        for j, agent_idx in enumerate(ids[i]):
            reward_history[agent_idx].append(rewards[j])
        if train:
            alives = env.get_alive(player_handles[i])
            # store samples in replay buffer (non-blocking)
            models[i].sample_step(rewards, alives, block=False)
        s = sum(rewards) / len(rewards)
        step_reward.append(s)
        total_reward[i] += s

        # compliant
        i = 0
        boycotting_ratio = 0
        reg_max = 3  # maximum number of apples to be collected
        rewards = env.get_reward(player_handles[i])
        def_reward = 0
        def_num = 0
        for def_idx in ids[1]:
            if len([1 for r in reward_history[def_idx] if r > reg_max]) > 0:
                # this agent behaves defectively
                def_num += 1
                def_reward += sum(reward_history[def_idx])
        if def_num > 0:  # boycotting reward shaping: penalize agent for success of the defective agents
            rewards = np.array([r - boycotting_ratio * (def_reward / def_num) for r in rewards])
        for j, agent_idx in enumerate(ids[i]):
            reward_history[agent_idx].append(rewards[j])
        if train:
            alives = env.get_alive(player_handles[i])
            # store samples in replay buffer (non-blocking)
            models[i].sample_step(rewards, alives, block=False)
        s = sum(rewards) / len(rewards)
        step_reward.append(s)
        total_reward[i] += s

        # render
        if render:
            env.render()

        # clear dead agents
        env.clear_dead()

        # respawn
        food_num = env.get_num(food_handle)
        for _ in range(5 - food_num):
            occupied_pos = cur_pos[0].tolist() + cur_pos[1].tolist() + env.get_pos(food_handle).tolist()

            pos = [random.randint(1, map_size - 2), random.randint(1, map_size - 2)]
            while pos in occupied_pos:
                pos = [random.randint(1, map_size - 2), random.randint(1, map_size - 2)]

            env.add_agents(food_handle, method="custom", pos=[pos])

        # check 'done' returned by 'sample' command
        if train:
            for model in models:
                model.check_done()

        if step_ct % print_every == 0:
            print("step %3d,  reward: %s,  total_reward: %s " %
                  (step_ct, np.around(step_reward, 2), np.around(total_reward, 2)))
        step_ct += 1
        if step_ct > 250:
            break

    sample_time = time.time() - start_time
    print("steps: %d,  total time: %.2f,  step average %.2f" % (step_ct, sample_time, sample_time / step_ct))

    # train
    total_loss, value = [0 for _ in range(n)], [0 for _ in range(n)]
    if train:
        print("===== train =====")
        start_time = time.time()

        # train models in parallel
        for i in range(n):
            models[i].train(print_every=2000, block=False)
        for i in range(n):
            total_loss[i], value[i] = models[i].fetch_train()

        train_time = time.time() - start_time
        print("train_time %.2f" % train_time)

    return python.magent.round(total_loss), python.magent.round(total_reward), python.magent.round(value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--render_every", type=int, default=100)
    parser.add_argument("--n_round", type=int, default=500)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--load_from", type=int)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--map_size", type=int, default=10)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--name", type=str, default="regulation_enf")
    args = parser.parse_args()

    # set logger
    python.magent.utility.init_logger(args.name)

    # init the game
    env = magent.GridWorld("regulation_enf", map_size=args.map_size)
    env.set_render_dir("build/render")

    # groups of agents
    handles = env.get_handles()
    food_handle = handles[0]
    player_handles = handles[1:]

    # load models
    names = ["compliant", "defective"]
    models = []
    for i in range(len(player_handles)):
        models.append(magent.ProcessingModel(
            env, player_handles[i], names[i], 20000+i, 4000, DeepQNetwork,
            batch_size=512, memory_size=2 ** 19,
            target_update=100, train_freq=4, use_dueling=True,
            use_double=True
        ))

    # load if
    savedir = 'save_model'
    if args.load_from is not None:
        start_from = args.load_from
        print("load ... %d" % start_from)
        for model in models:
            model.load(savedir, start_from)
    else:
        start_from = 0

    # print debug info
    print(args)
    print("view_space", env.get_view_space(player_handles[0]))
    print("feature_space", env.get_feature_space(player_handles[0]))

    # play
    start = time.time()
    for k in range(start_from, start_from + args.n_round):
        tic = time.time()
        eps = magent.utility.piecewise_decay(k, [0, 200, 400], [1, 0.2, 0.05]) if not args.greedy else 0

        loss, reward, value = play_a_round(env, args.map_size, food_handle, player_handles, models,
                                           print_every=50, train=args.train,
                                           render=args.render or (k + 1) % args.render_every == 0,
                                           eps=eps)  # for e-greedy
        log.info("round %d\t loss: %s\t reward: %s\t value: %s" % (k, loss, reward, value))
        print("round time %.2f  total time %.2f\n" % (time.time() - tic, time.time() - start))

        if (k + 1) % args.save_every == 0 and args.train:
            print("save model... ")
            for model in models:
                model.save(savedir, k)

    # send quit command
    for model in models:
        model.quit()
