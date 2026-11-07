from __future__ import annotations

from .models import JsonValue
from .task_models import BindingRole, TaskScope


def _channels(value: JsonValue) -> tuple[str, ...] | None:
    if not isinstance(value, list) or len(value) > 64:
        return None
    channels: list[str] = []
    for channel in value:
        if not isinstance(channel, str) or not channel or len(channel) > 128 or any(char.isspace() for char in channel):
            return None
        channels.append(channel)
    return tuple(channels) if len(set(channels)) == len(channels) else None


def resolve_binding(config: JsonValue, scope: TaskScope, parent_channel_id: str | None = None) -> BindingRole:
    if not isinstance(config, dict) or config.get("enabled") is not True:
        return BindingRole.NONE
    if type(config.get("schema_version")) is not int or config["schema_version"] != 1:
        return BindingRole.NONE
    if config.get("profile_id") != scope.profile_id or config.get("platform") != scope.platform or scope.platform != "discord":
        return BindingRole.NONE
    inquiry = _channels(config.get("inquiry_channels"))
    coordinator = _channels(config.get("coordinator_channels"))
    if inquiry is None or coordinator is None or set(inquiry) & set(coordinator):
        return BindingRole.NONE
    selected = scope.channel_id
    if selected not in inquiry and selected not in coordinator:
        selected = parent_channel_id or ""
    if selected in inquiry:
        return BindingRole.INQUIRY
    if selected in coordinator:
        return BindingRole.COORDINATOR
    return BindingRole.NONE
