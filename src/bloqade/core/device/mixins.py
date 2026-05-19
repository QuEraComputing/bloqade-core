from dataclasses import dataclass

from qlam_core.common import AppContext
from qlam_core.auth.client import AuthClient


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
