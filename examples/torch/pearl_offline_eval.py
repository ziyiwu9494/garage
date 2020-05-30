import argparse

from dowel import StdOutput, logger, tabular
from metaworld.benchmarks import ML45

from garage.envs import GarageEnv, normalize
from garage.experiment import SnapshotConfig, LocalRunner, MetaEvaluator
from garage.experiment.task_sampler import EnvPoolSampler
import garage.torch.utils as tu


def evaluate(meta_train_dir,
             max_path_length,
             adapt_rollout_per_task,
             use_gpu):
    snapshot_config = SnapshotConfig(snapshot_dir=meta_train_dir,
                                     snapshot_mode='all',
                                     snapshot_gap=1)

    runner = LocalRunner(snapshot_config=snapshot_config)

    ml45_test_envs = [
        GarageEnv(normalize(ML45.from_task(task_name)))
        for task_name in ML45.get_test_tasks().all_task_names
    ]
    test_env_sampler = EnvPoolSampler(ml45_test_envs)

    runner.restore(meta_train_dir)

    tu.set_gpu_mode(use_gpu, gpu_id=0)
    if use_gpu:
        runner._algo.to()

    meta_evaluator = MetaEvaluator(
        test_task_sampler=test_env_sampler,
        max_path_length=max_path_length,
        n_test_tasks=test_env_sampler.n_tasks,
        n_exploration_traj=adapt_rollout_per_task,
        prefix='')

    meta_evaluator.evaluate(runner._algo)
    logger.log(tabular)


if __name__ == '__main__':
    logger.add_output(StdOutput())

    parser = argparse.ArgumentParser()
    parser.add_argument('folder')
    parser.add_argument('--use_gpu', default=True)

    args = parser.parse_args()

    # If snapshot is trained on GPU, use_gpu must be set to True
    use_gpu = args.use_gpu
    meta_train_dir = args.folder
    max_path_length = 150
    adapt_rollout_per_task = 10

    evaluate(meta_train_dir,
             max_path_length,
             adapt_rollout_per_task,
             use_gpu)
