"""Per-tenant overrides stored by admins (model chains, usage limits) applied on top of environment settings."""
from __future__ import annotations

import dataclasses

from config.settings import Settings, get_logger

log = get_logger("model_config")
_LIMITS = ("rate_limit_per_min", "global_rate_limit_per_min", "daily_budget_usd")


def apply_tenant_overrides(settings: Settings, tenant_id: str, repo=None) -> Settings:
    try:
        if repo is None:
            from database.repository import get_repository
            repo = get_repository(settings)
        over = {}
        for role, entries in repo.model_chains(tenant_id).items():
            if entries and hasattr(settings, f"{role}_models"):
                over[f"{role}_models"] = tuple(entries)
        policy = repo.get_policy(tenant_id)
        for key in _LIMITS:
            if key in policy:
                over[key] = type(getattr(settings, key))(policy[key])
        return dataclasses.replace(settings, **over) if over else settings
    except Exception:  # overrides are optional; never break a request
        log.exception("tenant overrides could not be applied")
        return settings
