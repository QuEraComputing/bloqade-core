from dataclasses import dataclass
from typing import Callable, TypeVar

from qlam_core.auth.client import AuthClient
from qlam_core.common import AppContext
from qlam_core.errors import APIError

T = TypeVar("T")


@dataclass(kw_only=True)
class AuthMixin:
    """Mixin that provides authentication helpers for qlam API clients.

    Manages an `AppContext` scoped to a qlam context name and ensures the
    client is authenticated before making API calls.

    Attributes:
        context_name (str): Name of the qlam context to use.
        qpu_mode (str | None): Explicit qlam QPU mode to use for API calls.
            When None, qlam-core resolves it from configuration.
    """

    context_name: str
    qpu_mode: str | None = None

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
