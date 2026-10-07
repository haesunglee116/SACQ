"""Explicit settings for the exported calibration adapter (standard library only)."""
import math

TRAINING_KEYS = frozenset({
    'limit', 'epochs', 'global_batch', 'codebook_lr', 'scale_lr', 'val_every',
    'steps_per_invocation', 'fp32_codebook_grad', 'resume', 'expected_updates',
})


def validate_training(training, records_count):
    if not isinstance(training, dict) or set(training) != TRAINING_KEYS:
        raise ValueError(f'Training must explicitly contain exactly {sorted(TRAINING_KEYS)}')
    for key in ('limit', 'epochs', 'global_batch', 'val_every', 'expected_updates'):
        if type(training[key]) is not int or training[key] <= 0:
            raise ValueError(f'{key} must be a positive integer')
    if type(training['steps_per_invocation']) is not int or training['steps_per_invocation'] < 0:
        raise ValueError('steps_per_invocation must be a nonnegative integer')
    for key in ('codebook_lr', 'scale_lr'):
        value = training[key]
        if type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'{key} must be positive and finite')
    for key in ('fp32_codebook_grad', 'resume'):
        if type(training[key]) is not bool:
            raise ValueError(f'{key} must be a boolean')
    if training['limit'] != records_count:
        raise ValueError('Explicit request limit differs from supplied records')
    # Same positional split and hardware-independent budget as the trainer.
    validation_count = len(range(0, records_count, training['val_every']))
    train_count = records_count - validation_count
    if train_count <= 0 or train_count % training['global_batch']:
        raise ValueError('Training records must be nonempty and divisible by global_batch')
    budget = training['epochs'] * (train_count // training['global_batch'])
    if training['expected_updates'] > budget:
        raise ValueError('Expected completion exceeds the declared epoch budget')
    if training['steps_per_invocation'] == 0 and training['expected_updates'] != budget:
        raise ValueError('Without an early stop, expected completion must equal the epoch budget')
    return dict(train_records=train_count, validation_records=validation_count,
                epoch_budget=budget)


def calibration_argv(job):
    """Generate only supported original trainer flags; never fall back to defaults."""
    training = job['training']
    validate_training(training, len(job['all_records']))
    args = [job['trainer'], '--suite', job['suite'], '--limit', str(training['limit']),
            '--epochs', str(training['epochs']), '--global-batch', str(training['global_batch']),
            '--out-tag', job['tag'], '--alloc', job['allocation'],
            '--lr', str(training['codebook_lr']), '--lr-scale', str(training['scale_lr']),
            '--val-every', str(training['val_every']),
            '--steps', str(training['steps_per_invocation'])]
    if training['fp32_codebook_grad']:
        args.append('--fp32-codebook-grad')
    if training['resume']:
        args.append('--resume')
    return args


INDEX_KEYS=frozenset({'damping','block_columns','noise_seed_base','moment_chunk_rows',
                      'torch_num_threads','allow_tf32_matmul','allow_tf32_cudnn'})


def validate_index(settings):
    if not isinstance(settings,dict) or set(settings)!=INDEX_KEYS:
        raise ValueError(f'Index settings must explicitly contain exactly {sorted(INDEX_KEYS)}')
    for key in ('block_columns','moment_chunk_rows','torch_num_threads'):
        if type(settings[key]) is not int or settings[key]<=0:
            raise ValueError(f'{key} must be a positive integer')
    if settings['block_columns']%64:
        raise ValueError('block_columns must be a multiple of the supported group size 64')
    if type(settings['noise_seed_base']) is not int or settings['noise_seed_base']<0:
        raise ValueError('noise_seed_base must be an explicit nonnegative integer')
    damping=settings['damping']
    if type(damping) not in (int,float) or not math.isfinite(damping) or damping<=0:
        raise ValueError('damping must be positive and finite')
    for key in ('allow_tf32_matmul','allow_tf32_cudnn'):
        if type(settings[key]) is not bool:raise ValueError(f'{key} must be an explicit boolean')
    return settings
