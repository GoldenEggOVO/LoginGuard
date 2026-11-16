import numpy as np
import pandas as pd

from .generate_data import START


def generate_independent(seed=241):
    rng = np.random.default_rng(seed)
    rows = []

    def add(time, user, source, device, status, kind='normal', episode=None, step=None):
        rows.append((time, f'user_{user:02d}', source, device, status, kind, episode, step))


    for day in range(14):
        weekend = day % 7 in (5, 6)
        for user in range(24):
            rate = 7 if user < 16 else 6 if user < 21 else 12
            count = rng.poisson(rate * (.7 if weekend and user < 21 else 1))
            for _ in range(count):
                hour = float(rng.uniform(0, 24) if user >= 21 else rng.normal(13 if user < 16 else 2, 3)) % 24
                seconds = min(int(hour * 3600), 86300)
                time = START + pd.Timedelta(days=day, seconds=seconds)
                source = f'198.51.100.{100 + user % 3}' if rng.random() < .3 else f'192.0.2.{user + 10}'
                device = f'independent_{user}_{int(rng.choice([0, 1], p=[.85, .15]))}'
                if rng.random() < .12:
                    for j in range(int(rng.integers(1, 4))):
                        add(time + pd.Timedelta(seconds=j * 9), user, source, device, 'failure')
                    add(time + pd.Timedelta(seconds=30), user, source, device, 'success')
                else:
                    add(time, user, source, device, 'failure' if rng.random() < .05 else 'success')


    for day in range(10, 14):
        for batch, kind in enumerate(('brute_force', 'password_spray', 'suspicious_success')):
            user = int(rng.integers(24))
            start = START + pd.Timedelta(days=day, hours=int(rng.integers(4, 20)), minutes=int(rng.integers(60)))
            count = int(rng.integers(4, 8))
            familiar = rng.random() < .55
            source = f'198.51.100.{100 + user % 3}' if rng.random() < .5 else f'203.0.113.{day * 3 + batch}'
            elapsed = 0
            for j in range(count):
                target = (user + j * 2) % 24 if kind == 'password_spray' else user
                device = f'independent_{target}_0' if familiar else f'unknown_alt_{day}_{batch}'
                status = 'success' if kind == 'suspicious_success' and j == count - 1 else 'failure'
                add(start + pd.Timedelta(seconds=elapsed), target, source, device, status, kind, f'{day}_{batch}', j)
                elapsed += int(rng.integers(35, 181))

    frame = pd.DataFrame(rows, columns=['timestamp', 'user_id', 'ip_address', 'device_id', 'status',
                                       'attack_type', 'episode_id', 'episode_step'])
    frame = frame.sort_values('timestamp', kind='stable').reset_index(drop=True)
    frame.insert(0, 'event_id', np.arange(1, len(frame) + 1))
    labels = frame[['event_id', 'attack_type']].copy()
    labels.insert(1, 'is_anomaly', labels.attack_type.ne('normal').astype(int))
    episodes = frame[['event_id', 'episode_id', 'episode_step']].copy()
    return frame.drop(columns=['attack_type', 'episode_id', 'episode_step']), labels, episodes
