from hotshift.policies.adaptive import AdaptivePolicy
from hotshift.policies.base import CachePolicy
from hotshift.policies.decayed_lfu import DecayedLFUPolicy
from hotshift.policies.lru import LRUPolicy
from hotshift.policies.no_cache import NoCachePolicy
from hotshift.policies.static import StaticPolicy


def create_policy(name: str) -> CachePolicy:
    policies: dict[str, type[CachePolicy]] = {
        "static": StaticPolicy,
        "lru": LRUPolicy,
        "decayed_lfu": DecayedLFUPolicy,
        "adaptive": AdaptivePolicy,
        "no_cache": NoCachePolicy,
    }
    try:
        return policies[name]()
    except KeyError as error:
        raise ValueError(f"unknown cache policy: {name}") from error

