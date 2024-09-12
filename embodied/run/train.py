import re
from collections import defaultdict
from functools import partial as bind

import embodied
import numpy as np
import torch
import torch.nn.functional as F


def random_crop(imgs, output_size):
    """
    Randomly crop the input images.
    Args:
        imgs: Input images (batch_size, channels, height, width)
        output_size: Desired output size (height, width)
    """
    batch_size, _, height, width = imgs.shape
    crop_h, crop_w = output_size
    cropped_imgs = []
    for img in imgs:
        top = np.random.randint(0, height - crop_h + 1)
        left = np.random.randint(0, width - crop_w + 1)
        cropped_imgs.append(img[:, top:top + crop_h, left:left + crop_w])
    return torch.stack(cropped_imgs)


def info_nce_loss(features, batch_size, temperature=0.1):
    """
    Compute the InfoNCE loss for contrastive learning.
    Args:
        features: Normalized feature projections from the encoder (2*batch_size, projection_dim)
        batch_size: Size of the input batch
        temperature: Temperature scaling for the logits
    """
    labels = torch.cat([torch.arange(batch_size) for _ in range(2)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().to(features.device)

    features = F.normalize(features, dim=1)
    similarity_matrix = torch.matmul(features, features.T)

    # Mask out self-similarities
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(features.device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

    # Positive and negative pairs
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    # Compute logits
    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(features.device)

    return F.cross_entropy(logits / temperature, labels)


def train(make_agent, make_replay, make_env, make_logger, args):

    agent = make_agent()
    replay = make_replay()
    logger = make_logger()

    logdir = embodied.Path(args.logdir)
    logdir.mkdir()
    print('Logdir', logdir)
    step = logger.step
    usage = embodied.Usage(**args.usage)
    agg = embodied.Agg()
    epstats = embodied.Agg()
    episodes = defaultdict(embodied.Agg)
    policy_fps = embodied.FPS()
    train_fps = embodied.FPS()

    batch_steps = args.batch_size * (args.batch_length - args.replay_context)
    should_expl = embodied.when.Until(args.expl_until)
    should_train = embodied.when.Ratio(args.train_ratio / batch_steps)
    should_log = embodied.when.Clock(args.log_every)
    should_eval = embodied.when.Clock(args.eval_every)
    should_save = embodied.when.Clock(args.save_every)

    @embodied.timer.section('log_step')
    def log_step(tran, worker):

        episode = episodes[worker]
        episode.add('score', tran['reward'], agg='sum')
        episode.add('length', 1, agg='sum')
        episode.add('rewards', tran['reward'], agg='stack')

        if tran['is_first']:
            episode.reset()

        if worker < args.log_video_streams:
            for key in args.log_keys_video:
                if key in tran:
                    episode.add(f'policy_{key}', tran[key], agg='stack')
        for key, value in tran.items():
            if re.match(args.log_keys_sum, key):
                episode.add(key, value, agg='sum')
            if re.match(args.log_keys_avg, key):
                episode.add(key, value, agg='avg')
            if re.match(args.log_keys_max, key):
                episode.add(key, value, agg='max')

        if tran['is_last']:
            result = episode.result()
            logger.add({
                'score': result.pop('score'),
                'length': result.pop('length'),
            }, prefix='episode')
            rew = result.pop('rewards')
            if len(rew) > 1:
                result['reward_rate'] = (np.abs(rew[1:] - rew[:-1]) >= 0.01).mean()
            epstats.add(result)

    fns = [bind(make_env, i) for i in range(args.num_envs)]
    driver = embodied.Driver(fns, args.driver_parallel)
    driver.on_step(lambda tran, _: step.increment())
    driver.on_step(lambda tran, _: policy_fps.step())
    driver.on_step(replay.add)
    driver.on_step(log_step)

    dataset_train = iter(agent.dataset(bind(
        replay.dataset, args.batch_size, args.batch_length)))
    dataset_report = iter(agent.dataset(bind(
        replay.dataset, args.batch_size, args.batch_length_eval)))
    carry = [agent.init_train(args.batch_size)]
    carry_report = agent.init_report(args.batch_size)

    def train_step(tran, worker):
        if len(replay) < args.batch_size or step < args.train_fill:
            return
        for _ in range(should_train(step)):
            with embodied.timer.section('dataset_next'):
                batch = next(dataset_train)

            # Data augmentation: Random crop
            augmented_obs = random_crop(batch['obs'], output_size=(84, 84))  # Example output size
            latent, projection = agent.encoder(augmented_obs)

            # Contrastive loss
            contrastive_loss = info_nce_loss(projection, args.batch_size)

            # Reconstruction loss
            reconstructed = agent.decoder(latent)
            recon_loss = agent.decoder.reconstruction_loss(batch['obs'], reconstructed)

            # Other losses
            dynamics_loss = agent.compute_dynamics_loss(batch)
            policy_loss = agent.compute_policy_loss(batch)

            # Combine losses
            total_loss = (policy_loss +
                          args.lambda1 * dynamics_loss +
                          args.lambda2 * contrastive_loss +
                          args.lambda3 * recon_loss)

            # Backpropagate and update parameters
            outs, carry[0], mets = agent.train_step(total_loss, carry[0])
            train_fps.step(batch_steps)
            if 'replay' in outs:
                replay.update(outs['replay'])
            agg.add(mets, prefix='train')

    driver.on_step(train_step)

    checkpoint = embodied.Checkpoint(logdir / 'checkpoint.ckpt')
    checkpoint.step = step
    checkpoint.agent = agent
    checkpoint.replay = replay
    if args.from_checkpoint:
        checkpoint.load(args.from_checkpoint)
    checkpoint.load_or_save()
    should_save(step)  # Register that we just saved.

    print('Start training loop')
    policy = lambda *args: agent.policy(
        *args, mode='explore' if should_expl(step) else 'train')
    driver.reset(agent.init_policy)
    while step < args.steps:

        driver(policy, steps=10)

        if should_eval(step) and len(replay):
            mets, _ = agent.report(next(dataset_report), carry_report)
            logger.add(mets, prefix='report')

        if should_log(step):
            logger.add(agg.result())
            logger.add(epstats.result(), prefix='epstats')
            logger.add(embodied.timer.stats(), prefix='timer')
            logger.add(replay.stats(), prefix='replay')
            logger.add(usage.stats(), prefix='usage')
            logger.add({'fps/policy': policy_fps.result()})
            logger.add({'fps/train': train_fps.result()})
            logger.write()

        if should_save(step):
            checkpoint.save()

    logger.close()
