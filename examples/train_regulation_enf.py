import argparse
import time
import logging as log

from python import magent
from python.magent.builtin.tf_model import DeepQNetwork

def play_a_round(env, map_size, handles, models, print_every, train=True, render=False, eps=None):
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_every", type=int, default=2)
    parser.add_argument("--render_every", type=int, default=10)
    parser.add_argument("--n_round", type=int, default=500)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--load_from", type=int)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--map_size", type=int, default=1000)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--name", type=str, default="pursuit")
    args = parser.parse_args()

    # set logger
    magent.utility.init_logger(args.name)

    # init the game
    env = magent.GridWorld("regulation_enf", map_size=args.map_size)
    env.set_render_dir("build/render")

    # two groups of agents
    handles = env.get_handles()

    # load models
    names = ["compliant", "defective"]
    models = []
    for i in range(len(names)):
        models.append(magent.ProcessingModel(
            env, handles[i], names[i], 20000+i, 4000, DeepQNetwork,
            batch_size=512, memory_size=2 ** 22,
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
    print("view_space", env.get_view_space(handles[0]))
    print("feature_space", env.get_feature_space(handles[0]))

    # play
    start = time.time()
    for k in range(start_from, start_from + args.n_round):
        tic = time.time()
        eps = magent.utility.piecewise_decay(k, [0, 200, 400], [1, 0.2, 0.05]) if not args.greedy else 0

        loss, reward, value = play_a_round(env, args.map_size, handles, models,
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
