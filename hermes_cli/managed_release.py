from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import stat


MANAGED_RELEASE_ID = 'fork-b200-release-v1'
SOURCE_ROOT = Path(__file__).resolve().parents[1]
ACTIVATION_RECORD = Path('.hermes/managed-release.json')


@dataclass(frozen=True, slots=True)
class ManagedUpdateReceipt:
    schema_version: int
    release_id: str
    operation: str
    reason_code: str
    activation_state: str
    mutation_allowed: bool = False


class ManagedReleaseUpdateBlocked(RuntimeError):
    def __init__(self, receipt: ManagedUpdateReceipt):
        self.receipt = receipt
        super().__init__('Managed runtime update refused; use the governed release workflow.')


def require_unmanaged_update(root: Path, operation: str) -> None:
    root = Path(root).resolve()
    contract = root / 'hermes_cli/managed_release.py'
    try:
        managed = root == SOURCE_ROOT or contract.lstat() is not None
    except FileNotFoundError:
        managed = False
    except OSError:
        managed = True
    if not managed:
        return
    state = 'missing'
    try:
        path = root / ACTIVATION_RECORD
        record_stat = path.lstat()
        if not stat.S_ISREG(record_stat.st_mode) or record_stat.st_size > 65536:
            state = 'malformed'
        else:
            data = json.loads(path.read_text(encoding='utf-8'))
            state = ('present' if isinstance(data, dict) and type(data.get('schema_version')) is int
                     and data['schema_version'] == 1 and data.get('release_id') == MANAGED_RELEASE_ID
                     else 'malformed')
    except FileNotFoundError:
        pass
    except (OSError, UnicodeError, ValueError):
        state = 'malformed'
    raise ManagedReleaseUpdateBlocked(ManagedUpdateReceipt(
        1, MANAGED_RELEASE_ID, operation, 'managed_release_requires_governed_upgrade', state))
