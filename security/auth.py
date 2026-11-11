"""
Authentication and role-based access control.

    AUTH_MODE=none       single local user (owner of tenant "default"); 1.x behaviour
    AUTH_MODE=password   one shared APP_PASSWORD (1.x), still a single tenant
    AUTH_MODE=users      accounts with roles, tenants, API tokens, lockout, audit

Passwords: PBKDF2-HMAC-SHA256, 600,000 iterations, 16-byte salt (OWASP 2023+ guidance); never stored or
logged in plaintext. Lockout: 5 failed logins → 15 minutes. Password policy: ≥ 10 characters with letters
and digits. The first admin is created from ADMIN_USERNAME / ADMIN_PASSWORD (or scripts/create_admin.py).

Roles → permissions:
    admin   everything in its tenant (users, policies, models, all documents, audit log)
    editor  upload documents, delete/relabel its own documents, ask questions
    viewer  ask questions about documents its roles may read
"""
from __future__ import annotations

import hashlib
import hmac
import re
import secrets
import time
from typing import Optional, Sequence

from config.settings import Settings, get_logger
from core.service import Principal
from database.models import ROLES

log = get_logger("auth")

ITERATIONS = 600_000
MAX_FAILURES = 5
LOCKOUT_S = 15 * 60
PERMISSIONS = {
    "admin": {"ask", "read", "upload", "delete_own", "delete_any", "manage_users", "manage_policy", "manage_models",
              "view_audit", "run_evaluation", "relabel"},
    "editor": {"ask", "read", "upload", "delete_own", "relabel", "run_evaluation"},
    "viewer": {"ask", "read"},
}
_USERNAME = re.compile(r"^[a-z0-9._@-]{3,64}$")


class AuthError(Exception):
    def __init__(self, message: str):
        super().__init__(message)
        self.user_message = message


def hash_password(password: str, iterations: int = ITERATIONS) -> str:
    salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iterations)
    return f"pbkdf2_sha256${iterations}${salt.hex()}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, iters, salt_hex, dk_hex = stored.split("$")
        if algo != "pbkdf2_sha256":
            return False
        dk = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt_hex), int(iters))
        return hmac.compare_digest(dk.hex(), dk_hex)
    except (ValueError, AttributeError):
        return False


def check_password_policy(password: str) -> None:
    if len(password or "") < 10 or not re.search(r"[A-Za-z]", password) or not re.search(r"\d", password):
        raise AuthError("Passwords need at least 10 characters, including letters and digits.")


def has_permission(principal: Principal, permission: str) -> bool:
    return any(permission in PERMISSIONS.get(r, set()) for r in principal.roles)


def require(principal: Principal, permission: str) -> None:
    if not has_permission(principal, permission):
        raise PermissionError(f"'{permission}' is not allowed for roles {','.join(principal.roles)}")


def principal_for(user) -> Principal:
    return Principal(user.id, user.tenant_id, tuple(user.role_list), user.username)


class AuthService:
    def __init__(self, settings: Settings, repository=None):
        from database.repository import get_repository

        self.settings = settings
        self.repo = repository or get_repository(settings)

    def bootstrap_admin(self) -> Optional[str]:
        """Create the first admin from ADMIN_USERNAME/ADMIN_PASSWORD when no user exists yet."""
        s = self.settings
        if self.repo.count_users() or not (s.admin_username and s.admin_password):
            return None
        user = self.create_user(s.admin_username, s.admin_password, ["admin"], "default")
        log.info("bootstrap admin created")
        return user.id

    def create_user(self, username: str, password: str, roles: Sequence[str], tenant_id: str,
                    actor: Optional[Principal] = None):
        username = (username or "").strip().lower()
        if not _USERNAME.match(username):
            raise AuthError("Usernames are 3-64 characters: letters, digits, . _ @ -")
        if not roles or any(r not in ROLES for r in roles):
            raise AuthError(f"Roles must be among {', '.join(ROLES)}.")
        if actor is not None:
            require(actor, "manage_users")
            if actor.tenant_id != tenant_id:
                raise PermissionError("admins can only create users in their own tenant")
        check_password_policy(password)
        if self.repo.get_user_by_name(username):
            raise AuthError("This username is taken.")
        user = self.repo.create_user(tenant_id, username, hash_password(password), roles)
        self.repo.audit(tenant_id, actor.user_id if actor else "system", "user_created", user.id,
                        {"roles": list(roles)})
        return user

    def authenticate(self, username: str, password: str) -> Principal:
        user = self.repo.get_user_by_name(username)
        now = time.time()
        if user is None:
            verify_password(password or "", hash_password("timing-equaliser", 1000))  # similar work on misses
            raise AuthError("Invalid username or password.")
        if not user.active:
            raise AuthError("This account is disabled.")
        if user.locked_until and user.locked_until > now:
            raise AuthError("Too many failed attempts. Try again later.")
        if not verify_password(password or "", user.password_hash):
            fails = (user.failed_logins or 0) + 1
            self.repo.update_user(user.id, failed_logins=fails,
                                  locked_until=now + LOCKOUT_S if fails >= MAX_FAILURES else 0.0)
            self.repo.audit(user.tenant_id, user.id, "login_failed", user.username, {"failures": fails})
            raise AuthError("Invalid username or password.")
        self.repo.update_user(user.id, failed_logins=0, locked_until=0.0, last_login=now)
        self.repo.audit(user.tenant_id, user.id, "login", user.username)
        return principal_for(user)

    def principal_from_token(self, token: str) -> Optional[Principal]:
        user = self.repo.resolve_token(token)
        return principal_for(user) if user else None

    def issue_token(self, principal: Principal, name: str = "api", ttl_s: float = 30 * 86400) -> str:
        token = self.repo.create_token(principal.user_id, name, ttl_s)
        self.repo.audit(principal.tenant_id, principal.user_id, "token_issued", name)
        return token

    def change_password(self, principal: Principal, old: str, new: str) -> None:
        user = self.repo.get_user(principal.user_id)
        if user is None or not verify_password(old, user.password_hash):
            raise AuthError("The current password is wrong.")
        check_password_policy(new)
        self.repo.update_user(user.id, password_hash=hash_password(new))
        self.repo.revoke_tokens(user.id)
        self.repo.audit(user.tenant_id, user.id, "password_changed", user.username)

    def set_roles(self, actor: Principal, user_id: str, roles: Sequence[str], active: Optional[bool] = None) -> None:
        require(actor, "manage_users")
        user = self.repo.get_user(user_id)
        if user is None or user.tenant_id != actor.tenant_id:
            raise PermissionError("user not found in your tenant")
        if any(r not in ROLES for r in roles):
            raise AuthError("unknown role")
        fields = {"roles": list(roles)}
        if active is not None:
            fields["active"] = active
        self.repo.update_user(user_id, **fields)
        if active is False:
            self.repo.revoke_tokens(user_id)
        self.repo.audit(actor.tenant_id, actor.user_id, "user_updated", user_id, {"roles": list(roles),
                                                                                   "active": active})
