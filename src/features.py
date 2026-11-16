from collections import defaultdict, deque
import math
from numbers import Real
import pandas as pd

FEATURE_COLUMNS = ['hour', 'success', 'account_failures_60s', 'ip_failures_60s',
                   'ip_distinct_users_10m', 'new_device', 'prior_consecutive_failures']


def build_features(events, failure_gap_seconds=None):
    """Use current/past log records only; the optional timeout limits inactivity gaps."""
    if failure_gap_seconds is not None and (isinstance(failure_gap_seconds, bool)
            or not isinstance(failure_gap_seconds, Real) or not math.isfinite(failure_gap_seconds)
            or failure_gap_seconds <= 0):
        raise ValueError('failure_gap_seconds must be a finite positive number or None')
    times = pd.to_datetime(events['timestamp'])
    if not times.is_monotonic_increasing:
        raise ValueError('Events must be sorted by timestamp')
    account_failures, ip_failures, ip_users = defaultdict(deque), defaultdict(deque), defaultdict(deque)
    devices, streak = defaultdict(set), defaultdict(int)
    last_failure = {}
    result = []
    for row, ts in zip(events.itertuples(index=False), times):
        user, ip = row.user_id, row.ip_address
        af, ipf, recent = account_failures[user], ip_failures[ip], ip_users[ip]
        for q, duration in ((af, 60), (ipf, 60), (recent, 600)):
            while q and (ts - (q[0][0] if q is recent else q[0])).total_seconds() > duration:
                q.popleft()
        if (failure_gap_seconds is not None and user in last_failure
                and (ts - last_failure[user]).total_seconds() > failure_gap_seconds):
            streak[user] = 0
        prior = streak[user]
        success = int(row.status == 'success')
        if not success:
            af.append(ts)
            ipf.append(ts)
            streak[user] += 1
            last_failure[user] = ts
        else:
            streak[user] = 0
            last_failure.pop(user, None)
        recent.append((ts, user))
        new_device = int(row.device_id not in devices[user])
        devices[user].add(row.device_id)
        result.append([ts.hour, success, len(af), len(ipf), len({v for _, v in recent}), new_device, prior])
    return pd.DataFrame(result, columns=FEATURE_COLUMNS)
