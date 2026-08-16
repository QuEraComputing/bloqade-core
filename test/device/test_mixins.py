import importlib

import pytest
from qlam_core.errors import APIError

from bloqade.core.device.mixins import AuthMixin

from .fixtures import remote

mixins_mod = importlib.import_module("bloqade.core.device.mixins")


def test_call_with_auth_refresh_returns_value_and_does_not_refresh_on_success(
    monkeypatch,
):
    auth = AuthMixin(context_name="ctx")
    fake = remote.FakeAuthClient()
    monkeypatch.setattr(mixins_mod, "AuthClient", lambda app_context: fake)

    assert auth.call_with_auth_refresh(lambda: "ok") == "ok"
    assert fake.calls == []


def test_call_with_auth_refresh_reraises_non_403_without_refresh(monkeypatch):
    auth = AuthMixin(context_name="ctx")
    fake = remote.FakeAuthClient()
    monkeypatch.setattr(mixins_mod, "AuthClient", lambda app_context: fake)

    def fn():
        raise APIError(message="server error", status_code=500)

    with pytest.raises(APIError) as excinfo:
        auth.call_with_auth_refresh(fn)

    assert excinfo.value.status_code == 500
    assert fake.calls == []


def test_call_with_auth_refresh_retries_after_successful_refresh(monkeypatch):
    auth = AuthMixin(context_name="ctx")
    fake = remote.FakeAuthClient(refresh_result={"qlam": True})
    monkeypatch.setattr(mixins_mod, "AuthClient", lambda app_context: fake)
    invocations = []

    def fn():
        invocations.append(None)
        if len(invocations) == 1:
            raise APIError(message="permission denied", status_code=403)
        return "refreshed"

    assert auth.call_with_auth_refresh(fn) == "refreshed"
    assert len(invocations) == 2
    assert [name for name, _ in fake.calls] == ["refresh_credentials"]


def test_call_with_auth_refresh_retries_after_successful_refresh_on_401(monkeypatch):
    auth = AuthMixin(context_name="ctx")
    fake = remote.FakeAuthClient(refresh_result={"qlam": True})
    monkeypatch.setattr(mixins_mod, "AuthClient", lambda app_context: fake)
    invocations = []

    def fn():
        invocations.append(None)
        if len(invocations) == 1:
            raise APIError(message="token expired", status_code=401)
        return "refreshed"

    assert auth.call_with_auth_refresh(fn) == "refreshed"
    assert len(invocations) == 2
    assert [name for name, _ in fake.calls] == ["refresh_credentials"]


def test_call_with_auth_refresh_reraises_when_refresh_yields_no_credentials(
    monkeypatch,
):
    auth = AuthMixin(context_name="ctx")
    fake = remote.FakeAuthClient(refresh_result={"qlam": False})
    monkeypatch.setattr(mixins_mod, "AuthClient", lambda app_context: fake)
    invocations = []

    def fn():
        invocations.append(None)
        raise APIError(message="permission denied", status_code=403)

    with pytest.raises(APIError) as excinfo:
        auth.call_with_auth_refresh(fn)

    assert excinfo.value.status_code == 403
    assert len(invocations) == 1
    assert [name for name, _ in fake.calls] == ["refresh_credentials"]


def test_call_with_auth_refresh_only_retries_once(monkeypatch):
    auth = AuthMixin(context_name="ctx")
    fake = remote.FakeAuthClient(refresh_result={"qlam": True})
    monkeypatch.setattr(mixins_mod, "AuthClient", lambda app_context: fake)
    invocations = []

    def fn():
        invocations.append(None)
        raise APIError(message="still denied", status_code=403)

    with pytest.raises(APIError) as excinfo:
        auth.call_with_auth_refresh(fn)

    assert excinfo.value.status_code == 403
    assert len(invocations) == 2
    assert [name for name, _ in fake.calls] == ["refresh_credentials"]


def test_current_user_id_uses_user_info_api(monkeypatch):
    user_info = remote.make_user_info()
    fake = remote.FakeAuthClient(user_info=user_info)
    auth = AuthMixin(context_name="ctx")
    monkeypatch.setattr(mixins_mod, "AuthClient", lambda app_context: fake)

    assert auth.current_user_id() == remote.DEFAULT_USER_ID
    assert fake.calls == [("get_user_info", {"provider": None})]


def test_current_user_id_rejects_user_info_without_user_id(monkeypatch):
    fake = remote.FakeAuthClient(user_info=remote.make_user_info(user_id=None))
    auth = AuthMixin(context_name="ctx")
    monkeypatch.setattr(mixins_mod, "AuthClient", lambda app_context: fake)

    with pytest.raises(RuntimeError, match="has no user_id"):
        auth.current_user_id()
