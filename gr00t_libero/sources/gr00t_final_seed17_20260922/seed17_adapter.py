"""Explicit official-state reset adapter; no state generation or policy defaults."""
import hashlib
import numpy as np


def state_sha(state):
    a = np.ascontiguousarray(state)
    return hashlib.sha256(str((a.shape, a.dtype)).encode() + a.tobytes()).hexdigest()


def observation_sha(obs):
    h = hashlib.sha256()
    for key, value in sorted(obs.items()):
        h.update(key.encode())
        if isinstance(value, str):
            h.update(value.encode())
        else:
            a = np.ascontiguousarray(value)
            h.update(str((a.shape, a.dtype)).encode())
            h.update(a.tobytes())
    return h.hexdigest()


def official_env_class(libero_class):
    class OfficialStateEnv(libero_class):
        def reset(self, seed=None, options=None):
            if not isinstance(options, dict) or set(options) != {
                'episode_id', 'initial_state', 'initial_state_sha256'
            }:
                raise ValueError('Explicit official episode/state/hash required')
            e = options['episode_id']
            if type(e) is not int or not 0 <= e < 50:
                raise ValueError('Episode outside approved official0-49')
            state = np.array(options['initial_state'], copy=True)
            if state_sha(state) != options['initial_state_sha256']:
                raise ValueError('Official state hash mismatch')
            if seed is not None:
                if seed != 17:
                    raise ValueError('Only approved environment seed17 accepted')
                self._env.seed(seed)
            # Reset controller/observables first. No dummy action or simulation step.
            self._env.reset()
            raw = self._env.set_init_state(state)
            if not np.array_equal(self._env.get_sim_state(), state):
                raise ValueError('Simulator state differs from official state')
            obs = self._process_observation(raw)
            return obs, {'success': self._env.check_success()}
    return OfficialStateEnv


def episode_batches(episode_ids, slots):
    """Explicit identity assignment; never derive an episode from slot RNG."""
    ids = list(episode_ids)
    if slots != 5 or sorted(ids) != list(range(50)):
        raise ValueError('Expected five slots and fifty approved episodes exactly once')
    return [[(slot, e) for slot, e in enumerate(ids[i:i+slots])]
            for i in range(0, len(ids), slots)]

