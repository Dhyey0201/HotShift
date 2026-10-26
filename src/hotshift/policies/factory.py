from hotshift.policies.adaptive import AdaptivePolicy
from hotshift.policies.base import CachePolicy
from hotshift.policies.decayed_lfu import DecayedLFUPolicy
from hotshift.policies.lru import LRUPolicy
from hotshift.policies.no_cache import NoCachePolicy
from hotshift.policies.static import StaticPolicy

POLICY_NAMES = ("no_cache", "static", "lru", "decayed_lfu", "adaptive")


def create_policy(name: str, row_bytes: int | None = None) -> CachePolicy:
    policies: dict[str, type[CachePolicy]] = {
        "static": StaticPolicy,
        "lru": LRUPolicy,
        "decayed_lfu": DecayedLFUPolicy,
        "adaptive": AdaptivePolicy,
        "no_cache": NoCachePolicy,
    }
    try:
        policy_type = policies[name]
    except KeyError as error:
        raise ValueError(f"unknown cache policy: {name}") from error
    if policy_type is AdaptivePolicy:
        return AdaptivePolicy(row_bytes=row_bytes)
    return policy_type()
