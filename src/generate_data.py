import numpy as np
import pandas as pd

START = pd.Timestamp('2024-01-01')


def generate(seed=42, scenario='baseline', *, test_profile=None, include_episodes=False):
    if scenario not in {'baseline', 'familiar_slow'}:
        raise ValueError(f'Unknown scenario: {scenario}')
    if test_profile is not None:
        if (scenario != 'baseline' or not isinstance(test_profile, tuple) or len(test_profile) != 3
                or test_profile[0] not in (20, 60, 150)
                or type(test_profile[1]) is not bool or type(test_profile[2]) is not bool):
            raise ValueError('test_profile must be (20|60|150, bool, bool) with baseline scenario')
    rng = np.random.default_rng(seed)
    users = [f'user_{i:02d}' for i in range(25)]
    rows = []

    def add(ts, user, ip, device, status, kind='normal', episode_id=None, episode_step=None):
        rows.append((pd.Timestamp(ts), user, ip, device, status, kind, episode_id, episode_step))


    for day in range(14):
        for _ in range(155):
            u = int(rng.integers(25))
            hour = int(rng.integers(7, 23) if rng.random() < .92 else rng.integers(0, 7))
            ts = START + pd.Timedelta(days=day, hours=hour, minutes=int(rng.integers(60)), seconds=int(rng.integers(60)))
            ip = '198.51.100.200' if rng.random() < .16 else f'192.0.2.{u + 1}'
            device = f'device_{u}_{int(rng.choice([0, 1, 2], p=[.86, .11, .03]))}'
            add(ts, users[u], ip, device, 'failure' if rng.random() < .10 else 'success')


    for day in range(7, 14):
        for batch in range(4):
            base = START + pd.Timedelta(days=day, hours=int(rng.integers(6, 23)), minutes=int(rng.integers(60)))
            u = int(rng.integers(25))
            kind = ['brute_force', 'password_spray', 'suspicious_success'][batch % 3]
            count = int(rng.integers(3, 9))
            for j in range(count):
                target = int((u + j) % 25) if kind == 'password_spray' else u
                status = 'success' if kind == 'suspicious_success' and j == count - 1 else 'failure'
                spacing = int(rng.integers(8, 30))
                episode_id = f'{day}_{batch}'
                if test_profile is not None and day >= 10:
                    gap, familiar, shared = test_profile
                    add(base + pd.Timedelta(seconds=j * gap), users[target],
                        '198.51.100.200' if shared else f'203.0.113.{1 + day * 4 + batch}',
                        f'device_{target}_0' if familiar else f'unknown_{day}_{batch}',
                        status, kind, episode_id, j)
                elif scenario == 'familiar_slow' and day >= 10:
                    add(base + pd.Timedelta(seconds=j * 150), users[target], '198.51.100.200',
                        f'device_{target}_0', status, kind, episode_id, j)
                else:
                    add(base + pd.Timedelta(seconds=j * spacing), users[target],
                        f'203.0.113.{1 + day * 4 + batch}', f'unknown_{day}_{batch}', status, kind, episode_id, j)

    df = pd.DataFrame(rows, columns=['timestamp', 'user_id', 'ip_address', 'device_id', 'status', 'attack_type', 'episode_id', 'episode_step'])
    df = df.sort_values('timestamp', kind='stable').reset_index(drop=True)
    df.insert(0, 'event_id', np.arange(1, len(df) + 1))
    labels = df[['event_id', 'attack_type']].copy()
    labels.insert(1, 'is_anomaly', (labels.attack_type != 'normal').astype(int))
    episodes = df[['event_id', 'episode_id', 'episode_step']].copy()
    events = df.drop(columns=['attack_type', 'episode_id', 'episode_step'])
    return (events, labels, episodes) if include_episodes else (events, labels)
