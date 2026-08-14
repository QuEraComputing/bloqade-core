import base64
import json
from dataclasses import dataclass
from typing import Callable, TypeVar
from uuid import UUID

from qlam_core.auth.client import AuthClient
from qlam_core.common import AppContext
from qlam_core.errors import APIError

T = TypeVar("T")


def _user_id_from_access_token(token: str) -> UUID | None:
    """Extract the QLAM user UUID from an OAuth access token, if present.

    The gateway identifies the caller through a namespaced `user_id` claim
    (e.g. `https://v2/dev/user_id`); the namespace prefix varies per
    deployment, so the claim is matched by its `user_id` suffix.

    Args:
        token (str): Encoded JWT access token.

    Returns:
        UUID | None: The user UUID, or None when the token has no readable
            user-id claim.
    """
    segments = token.split(".")
    if len(segments) != 3:
        return None

    payload = segments[1]
    # JWT segments are unpadded base64url; restore padding before decoding
    payload += "=" * (-len(payload) % 4)
    try:
        claims = json.loads(base64.urlsafe_b64decode(payload))
    except (ValueError, UnicodeDecodeError):
        return None

    if not isinstance(claims, dict):
        return None

    for key, value in claims.items():
        if key == "user_id" or key.endswith("/user_id"):
            try:
                return UUID(str(value))
            except ValueError:
                return None

    return None


@dataclass(kw_only=True)
class AuthMixin:
    """Mixin that provides authentication helpers for qlam API clients.

    Manages an `AppContext` scoped to a qlam context name and ensures the
    client is authenticated before making API calls.

    Attributes:
        context_name (str): Name of the qlam context to use.
    """

    context_name: str

    @property
    def app_context(self) -> AppContext:
        """The `AppContext` used to authenticate and connect to the backend.

        Returns:
            AppContext: An app context scoped to `context_name`.
        """
        return AppContext(self.context_name)

    def authenticate(self):
        """Ensure the client is authenticated, triggering a login if needed.

        Returns:
            The login result when a login is performed; otherwise None.
        """
        with AuthClient(self.app_context) as client:
            if client.is_authenticated():
                return

            return client.login()

    def current_user_id(self) -> UUID:
        """Return the authenticated user's QLAM user UUID.

        The UUID is read from the `user_id` claim of the current OAuth
        access token — the same claim the API gateway uses to identify the
        caller. Call `authenticate` first to ensure a credential exists.

        Returns:
            UUID: The authenticated user's ID.

        Raises:
            RuntimeError: If no credential carries a readable user-id claim.
        """
        with AuthClient(self.app_context) as client:
            for provider in client.list_providers():
                credential = client.get_credential(provider["name"])
                token = getattr(credential, "access_token", None)
                if token is None:
                    continue

                user_id = _user_id_from_access_token(token)
                if user_id is not None:
                    return user_id

        raise RuntimeError(
            "Could not determine the current user: no credential in context "
            f"{self.context_name!r} carries a user_id claim"
        )

    def call_with_auth_refresh(self, fn: Callable[[], T]) -> T:
        """Run a qlam API call, refreshing credentials once on a 401 or 403.

        If `fn` raises an `APIError` with status 401 (token expired) or 403
        (access denied), a best-effort non-interactive credential refresh is
        attempted via `AuthClient`. When the refresh updates at least one
        provider's credentials, `fn` is invoked again. Any other error, or a
        refresh that produces no fresh credentials, propagates the original
        exception.

        Only one retry is attempted; a second 401/403 is re-raised.

        Args:
            fn (Callable[[], T]): Zero-argument callable that performs the
                qlam API call.

        Returns:
            T: The value returned by `fn`.

        Raises:
            APIError: When `fn` raises an `APIError` whose status is neither
                401 nor 403, when refresh produces no fresh credentials, or
                when the retry also fails.
        """
        try:
            return fn()
        except APIError as e:
            if e.status_code not in (401, 403):
                raise
            with AuthClient(self.app_context) as client:
                refresh_results = client.refresh_credentials()
            if not any(refresh_results.values()):
                raise
            return fn()
