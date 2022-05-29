import argparse
import time
import logging as log

import numpy as np
import collections
import magent
from magent.builtin.tf_model import DeepQNetwork


def defective_detector(history, ids, old_rewards_checked):
    compliant_agents = []
    defective_agents = []
    defective = False
    for group in [0,1]:
        for idx, id in enumerate(ids[group]):
            for previous_reward in history[idx][-old_rewards_checked:]:
                if previous_reward > 3:
                    defective_agents.append(idx)
                    defective = True
                    break
            if not defective:
                compliant_agents.append(idx)
    return compliant_agents, defective_agents


def play_a_round(env, map_size, handles, player_handles, food_handles, models, print_every, train=True, render=False, eps=None):
    env.reset()

    # add 4 compliant and 1 defective agents and 3 apple trees on random places on the map
    env.add_agents(handles[0], method="random", n=num_trees)
    env.add_agents(handles[1], method="random", n=4)
    env.add_agents(handles[2], method="random", n=1)

    step_ct = 0
    done = False

    n = len(player_handles)

    history = collections.defaultdict(list)
    total_rewards = [0 for _ in range(n)]
    rewards = [None for _ in range(n)]
    alives = [None for _ in range(n)]
    prev_pos = [None for _ in range(n)]
    cur_pos  = [None for _ in range(n)]
    obs = [[] for _ in range(n)]
    ids = [[] for _ in range(n)]
    acts = [[] for _ in range(n)]
    nums = [env.get_num(handle) for handle in player_handles]
    total_reward = [0 for _ in range(n)]

    print("===== sample =====")
    print("eps %s number %s" % (eps, nums))
    start_time = time.time()

    #####
    # diminishing reward shaping config
    #####
    old_rewards_checked = 5
    thresh = 2
    ng = -3

    X_train = []
    y_train = []
    while not done:
        nums = [env.get_num(handle) for handle in player_handles]
        assert(nums == [4, 1])
        # get observation
        for i in range(n):
            obs[i] = env.get_observation(player_handles[i])
            ids[i] = env.get_agent_id(player_handles[i])
            prev_pos[i] = env.get_pos(player_handles[i])

        for i in [1, 0]:
            ##########
            # add custom feature
            ########
            # give 2D ID embedding
            cnt = 2

            if diminishing:
                for j in range(len(ids[i])):
                    obs[i][1][j, cnt] = sum(history[ids[i][j]][-old_rewards_checked+1:])
            cnt += 1


            food_positions = env.get_pos(food_handle).tolist()
            assert len(food_positions) == num_trees
            for food_pos in food_positions:
                for j in range(len(ids[i])):
                    obs[i][1][j, cnt] = food_pos[0]
                    obs[i][1][j, cnt+1] = food_pos[1]

                cnt += 2

            # add feature, add coordinate between agents
            for k in range(n):
                for l in range(len(ids[k])):
                    obs[i][1][:, cnt] = prev_pos[k][l][0]
                    obs[i][1][:, cnt+1] = prev_pos[k][l][1]
                    cnt += 2

            assert cnt == 19

            acts[i] = models[i].infer_action(obs[i], ids[i], policy='e_greedy', eps=eps, block=True)
            env.set_action(player_handles[i], acts[i])

        # simulate one step
        done = env.step()

        for i in range(n):
            rewards[i] = env.get_reward(player_handles[i])
            alives[i] = env.get_alive(player_handles[i])
            cur_pos[i] = env.get_pos(player_handles[i])
            total_rewards[i] += (np.array(rewards[i])).sum()

        if diminishing:
            for i in [0, 1]:
                compliant_agents, defective_agents = defective_detector(history, ids, old_rewards_checked)
                s = 0
                for defective_id in defective_agents:
                    s += reward[i][defective_id]
                for j in range(len(ids[i])):
                    idx = ids[i][j]
                    ori_reward = rewards[i][j]    # True reward of the agent, the rewards list contains the perceived reward
                    history[idx].append(int(ori_reward))

                    if i == 0 and len(defective_agents) > 0:  # We only apply the boycot function in compliant agents
                        new_reward = ori_reward - boycot_ratio * (s/len(defective_agents))
                        rewards[i][j] = new_reward

        # sample
        step_reward = []
        for i in range(n):
            if train:
                # store samples in replay buffer (non-blocking)
                models[i].sample_step(rewards[i], alives[i], block=False)
            s = sum(rewards)
            step_reward.append(s)
            total_reward[i] += s

        # render
        if render:
            env.render()

        # clear dead agents
        env.clear_dead()

        # check 'done' returned by 'sample' command
        if train:
            for model in models:
                model.check_done()

        # respawn
        food_num = env.get_num(food_handle)
        env.add_agents(food_handle, method="random", n=num_trees-food_num)

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

    return magent.round(total_loss), magent.round(total_reward), magent.round(value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--render_every", type=int, default=10)
    parser.add_argument("--n_round", type=int, default=1000)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--load_from", type=int)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--map_size", type=int, default=10)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--name", type=str, default="regulation_enf")
    args = parser.parse_args()

    diminishing = True
    boycot_ratio = 1
    num_trees = 3

    # set logger
    magent.utility.init_logger(args.name)

    # init the game
    env = magent.GridWorld("regulation_enf", map_size=args.map_size)
    env.set_render_dir("build/render")

    # groups of agents
    handles = env.get_handles()
    food_handle = handles[0]
    player_handles = handles[1:]

    # load models
    names = ["apple", "compliant", "defective"]
    models = []
    for i in range(1, len(names)):
        models.append(magent.ProcessingModel(
            env, handles[i], names[i], 20000+i, 4000, DeepQNetwork,
            batch_size=512, memory_size=2 ** 20,
            target_update=1000, train_freq=4, use_dueling=True,
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
    print("view_space", env.get_view_space(handles[1]))
    print("feature_space", env.get_feature_space(handles[1]))

    # play
    start = time.time()
    for k in range(start_from, start_from + args.n_round):
        tic = time.time()
        eps = magent.utility.piecewise_decay(k, [0, 200, 400], [1, 0.2, 0.05]) if not args.greedy else 0

        loss, reward, value = play_a_round(env, args.map_size, handles, player_handles, food_handle, models,
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
