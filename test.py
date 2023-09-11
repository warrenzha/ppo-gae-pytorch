import time
import utils as utils
import gym

from agent.ppo_discrete import make_ppo_discrete
from agent.ppo_continous import make_ppo_continous

from arguments import parse_args
from env.config import get_config



#################################### Testing ###################################
def evaluate(args):
    print("============================================================================================")

    ################## hyperparameters ##################
    config = get_config(args.env_name, args.seed)

    env_name = config.env_name
    args.has_continuous_action_space = config.has_continuous_action_space
    max_ep_len = config.max_ep_len
    action_std = config.action_std

    render = args.render          # render environment on screen
    frame_delay = 0               # if required; add delay b/w frames

    random_seed = config.random_seed  #### set this to load a particular checkpoint trained on random seed
    run_num_pretrained = args.pretrained  #### set this to load a particular checkpoint num
    total_test_episodes = config.total_test_episodes    # total num of testing episodes
    #####################################################

    env = gym.make(env_name)

    # state space dimension
    args.state_dim = env.observation_space.shape[0]

    has_continuous_action_space = config.has_continuous_action_space

    # action space dimension
    if args.has_continuous_action_space:
        args.action_dim = env.action_space.shape[0]
        args.max_action = float(env.action_space.high[0])
    else:
        args.action_dim = env.action_space.n

    # initialize a PPO agent
    if args.has_continuous_action_space:
        ppo_agent = make_ppo_continous(args)
    else:
        ppo_agent = make_ppo_discrete(args)
    state_norm = utils.Normalization(shape=args.state_dim)  # Trick 2:state normalization

    # preTrained weights directory
    directory = "PPO_preTrained" + '/' + env_name + '/'
    checkpoint_path_actor = directory + "PPO_Actor_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
    checkpoint_path_critic = directory + "PPO_Critic_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
    print("loading network from : " + checkpoint_path_actor + " & " + checkpoint_path_critic)

    ppo_agent.load(checkpoint_path_actor, checkpoint_path_critic)

    print("--------------------------------------------------------------------------------------------")

    test_running_reward = 0

    for ep in range(1, total_test_episodes+1):
        ep_reward = 0
        s = env.reset()
        if args.use_state_norm:  # During the evaluating,update=False
            s = state_norm(s, update=False)
        done = False

        while not done:
            a = ppo_agent.evaluate(s)  # We use the deterministic policy during the evaluating
            s_, r, done, _ = env.step(a)

            if args.use_state_norm:
                s_ = state_norm(s_, update=False)

            ep_reward += r
            s = s_

            if render:
                env.render()
                time.sleep(frame_delay)

        test_running_reward +=  ep_reward
        print('Episode: {} \t\t Reward: {}'.format(ep, round(ep_reward, 2)))

    env.close()

    print("============================================================================================")

    avg_test_reward = test_running_reward / total_test_episodes
    avg_test_reward = round(avg_test_reward, 2)
    print("average test reward : " + str(avg_test_reward))

    print("============================================================================================")


if __name__ == '__main__':
    args = parse_args()
    print(args)

    evaluate(args)