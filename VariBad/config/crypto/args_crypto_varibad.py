import argparse
from utils.helpers import boolean_argument


def get_args(rest_args):
    parser = argparse.ArgumentParser()

    # --- GENERAL ---

    # training parameters
    parser.add_argument('--num_frames', type=int, default=1e7, help='number of frames to train')
    parser.add_argument('--max_rollouts_per_task', type=int, default=1, help='Each episode is a new random slice')
    parser.add_argument('--exp_label', default='varibad_crypto', help='label')
    parser.add_argument('--env_name', default='CryptoPortfolio-v0', help='environment to train on')

    # --- POLICY ---

    # what to pass to the policy (note this is after the encoder)
    parser.add_argument('--pass_state_to_policy', type=boolean_argument, default=True, help='condition policy on state')
    parser.add_argument('--pass_latent_to_policy', type=boolean_argument, default=True, help='condition policy on VAE latent')
    # No Oracle belief/task for Crypto
    parser.add_argument('--pass_belief_to_policy', type=boolean_argument, default=False, help='condition policy on ground-truth belief')
    parser.add_argument('--pass_task_to_policy', type=boolean_argument, default=False, help='condition policy on ground-truth task description')

    # using separate encoders for the different inputs ("None" uses no encoder)
    parser.add_argument('--policy_state_embedding_dim', type=int, default=64)
    parser.add_argument('--policy_latent_embedding_dim', type=int, default=64)
    parser.add_argument('--policy_belief_embedding_dim', type=int, default=None)
    parser.add_argument('--policy_task_embedding_dim', type=int, default=None)

    # normalising (inputs/rewards/outputs)
    parser.add_argument('--norm_state_for_policy', type=boolean_argument, default=True, help='normalise state input')
    parser.add_argument('--norm_latent_for_policy', type=boolean_argument, default=True, help='normalise latent input')
    parser.add_argument('--norm_belief_for_policy', type=boolean_argument, default=True, help='normalise belief input')
    parser.add_argument('--norm_task_for_policy', type=boolean_argument, default=True, help='normalise task input')
    parser.add_argument('--norm_rew_for_policy', type=boolean_argument, default=True, help='normalise rew for RL train')
    parser.add_argument('--norm_actions_pre_sampling', type=boolean_argument, default=False, help='normalise policy output')
    parser.add_argument('--norm_actions_post_sampling', type=boolean_argument, default=False, help='normalise policy output')

    # network (Increased size for complex financial data)
    parser.add_argument('--policy_layers', nargs='+', default=[256, 256])
    parser.add_argument('--policy_activation_function', type=str, default='tanh', help='tanh/relu/leaky-relu')
    parser.add_argument('--policy_initialisation', type=str, default='normc', help='normc/orthogonal')
    parser.add_argument('--policy_anneal_lr', type=boolean_argument, default=False, help='anneal LR over time')

    # RL algorithm
    parser.add_argument('--policy', type=str, default='ppo', help='choose: a2c, ppo')
    parser.add_argument('--policy_optimiser', type=str, default='adam', help='choose: rmsprop, adam')

    # PPO specific
    parser.add_argument('--ppo_num_epochs', type=int, default=4, help='number of epochs per PPO update')
    parser.add_argument('--ppo_num_minibatch', type=int, default=8, help='number of minibatches to split the data')
    parser.add_argument('--ppo_use_huberloss', type=boolean_argument, default=True, help='use huberloss instead of MSE')
    parser.add_argument('--ppo_use_clipped_value_loss', type=boolean_argument, default=True, help='clip value loss')
    parser.add_argument('--ppo_clip_param', type=float, default=0.2, help='clamp param')

    # other hyperparameters
    parser.add_argument('--lr_policy', type=float, default=3e-4, help='learning rate')
    parser.add_argument('--num_processes', type=int, default=16,
                        help='number of parallel environments')
    parser.add_argument('--policy_num_steps', type=int, default=500,
                        help='steps per process before update')
    parser.add_argument('--policy_eps', type=float, default=1e-8, help='optimizer epsilon')
    parser.add_argument('--policy_init_std', type=float, default=1.0, help='only used for continuous actions')
    parser.add_argument('--policy_value_loss_coef', type=float, default=0.5, help='value loss coefficient')
    parser.add_argument('--policy_entropy_coef', type=float, default=0.01, help='entropy term coefficient')
    parser.add_argument('--policy_gamma', type=float, default=0.99, help='discount factor')
    parser.add_argument('--policy_use_gae', type=boolean_argument, default=True,
                        help='use generalized advantage estimation')
    parser.add_argument('--policy_tau', type=float, default=0.95, help='gae parameter')
    parser.add_argument('--use_proper_time_limits', type=boolean_argument, default=True,
                        help='treat timeout and death differently')
    parser.add_argument('--policy_max_grad_norm', type=float, default=0.5, help='max norm of gradients')
    parser.add_argument('--encoder_max_grad_norm', type=float, default=None, help='max norm of gradients')
    parser.add_argument('--decoder_max_grad_norm', type=float, default=None, help='max norm of gradients')

    # --- VAE TRAINING ---

    # general
    parser.add_argument('--lr_vae', type=float, default=0.001)
    parser.add_argument('--size_vae_buffer', type=int, default=500000,
                        help='how many trajectories to keep in VAE buffer')
    parser.add_argument('--precollect_len', type=int, default=2000,
                        help='how many frames to pre-collect')
    parser.add_argument('--vae_buffer_add_thresh', type=float, default=1,
                        help='probability of adding a new trajectory to buffer')
    parser.add_argument('--vae_batch_num_trajs', type=int, default=15,
                        help='how many trajectories to use for VAE update')
    parser.add_argument('--tbptt_stepsize', type=int, default=None,
                        help='stepsize for truncated BPTT')
    parser.add_argument('--vae_subsample_elbos', type=int, default=None,
                        help='subsample ELBOs')
    parser.add_argument('--vae_subsample_decodes', type=int, default=None,
                        help='subsample decodes')
    parser.add_argument('--vae_avg_elbo_terms', type=boolean_argument, default=False,
                        help='Average ELBO terms')
    parser.add_argument('--vae_avg_reconstruction_terms', type=boolean_argument, default=False,
                        help='Average reconstruction terms')
    parser.add_argument('--num_vae_updates', type=int, default=2,
                        help='how many VAE update steps to take per meta-iteration')
    parser.add_argument('--pretrain_len', type=int, default=0, help='pre-train VAE')
    parser.add_argument('--kl_weight', type=float, default=1.0, help='weight for the KL term')

    parser.add_argument('--split_batches_by_task', type=boolean_argument, default=False,
                        help='split batches by task')
    parser.add_argument('--split_batches_by_elbo', type=boolean_argument, default=False,
                        help='split batches by elbo')

    # - encoder
    parser.add_argument('--action_embedding_size', type=int, default=16)
    parser.add_argument('--state_embedding_size', type=int, default=32)
    parser.add_argument('--reward_embedding_size', type=int, default=16)
    parser.add_argument('--encoder_layers_before_gru', nargs='+', type=int, default=[])
    parser.add_argument('--encoder_gru_hidden_size', type=int, default=128, help='dimensionality of RNN hidden state')
    parser.add_argument('--encoder_layers_after_gru', nargs='+', type=int, default=[])
    parser.add_argument('--latent_dim', type=int, default=10, help='dimensionality of latent space')

    # - decoder: rewards (We want to predict rewards to learn profitability)
    parser.add_argument('--decode_reward', type=boolean_argument, default=True, help='use reward decoder')
    parser.add_argument('--normalise_rew_targets', type=boolean_argument, default=False, help='normalize reward targets')
    parser.add_argument('--rew_loss_coeff', type=float, default=1.0, help='weight for reward loss')
    parser.add_argument('--input_prev_state', type=boolean_argument, default=True, help='use prev state for rew pred')
    parser.add_argument('--input_action', type=boolean_argument, default=True, help='use prev action for rew pred')
    parser.add_argument('--reward_decoder_layers', nargs='+', type=int, default=[64, 32])
    parser.add_argument('--multihead_for_reward', type=boolean_argument, default=False,
                        help='one head per reward pred')
    parser.add_argument('--rew_pred_type', type=str, default='deterministic',
                        help='choose: deterministic')

    # - decoder: state transitions (Predicting market moves)
    parser.add_argument('--decode_state', type=boolean_argument, default=True, help='use state decoder')
    parser.add_argument('--state_loss_coeff', type=float, default=1.0, help='weight for state loss')
    parser.add_argument('--state_decoder_layers', nargs='+', type=int, default=[64, 32])
    parser.add_argument('--state_pred_type', type=str, default='deterministic', help='choose: deterministic')

    # - decoder: ground-truth task (NONE for Crypto)
    parser.add_argument('--decode_task', type=boolean_argument, default=False, help='use task decoder')
    parser.add_argument('--task_loss_coeff', type=float, default=1.0, help='weight for task loss')
    parser.add_argument('--task_decoder_layers', nargs='+', type=int, default=[64, 32])
    parser.add_argument('--task_pred_type', type=str, default='task_id', help='choose: task_id')

    # --- ABLATIONS ---
    parser.add_argument('--disable_decoder', type=boolean_argument, default=False,
                        help='train without decoder')
    parser.add_argument('--disable_stochasticity_in_latent', type=boolean_argument, default=False,
                        help='use auto-encoder')
    parser.add_argument('--disable_kl_term', type=boolean_argument, default=False,
                        help='dont use the KL regularising loss term')
    parser.add_argument('--decode_only_past', type=boolean_argument, default=False,
                        help='only decoder past observations')
    parser.add_argument('--kl_to_gauss_prior', type=boolean_argument, default=False,
                        help='KL term to fixed Gaussian prior')

    parser.add_argument('--rlloss_through_encoder', type=boolean_argument, default=True,
                        help='backprop rl loss through encoder (End-to-End training)')
    parser.add_argument('--add_nonlinearity_to_latent', type=boolean_argument, default=False,
                        help='Use relu before feeding latent to policy')
    parser.add_argument('--vae_loss_coeff', type=float, default=1.0,
                        help='weight for VAE loss')

    parser.add_argument('--sample_embeddings', type=boolean_argument, default=False,
                        help='sample embedding for policy')

    parser.add_argument('--disable_metalearner', type=boolean_argument, default=False)
    parser.add_argument('--single_task_mode', type=boolean_argument, default=False)

    # --- OTHERS ---

    parser.add_argument('--log_interval', type=int, default=25, help='log interval')
    parser.add_argument('--save_interval', type=int, default=500, help='save interval')
    parser.add_argument('--save_intermediate_models', type=boolean_argument, default=True, help='save all models')
    parser.add_argument('--eval_interval', type=int, default=50, help='eval interval')
    parser.add_argument('--vis_interval', type=int, default=500, help='vis interval')
    parser.add_argument('--results_log_dir', default=None, help='directory to save results')

    # general settings
    parser.add_argument('--seed',  nargs='+', type=int, default=[73])
    parser.add_argument('--deterministic_execution', type=boolean_argument, default=False,
                        help='Make code fully deterministic')

    return parser.parse_args(rest_args)
