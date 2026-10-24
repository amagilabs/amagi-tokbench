import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from fakeserver import FakeServer  # noqa: E402

FAKE_SMI = Path(__file__).parent / "fake_nvidia_smi.py"


@pytest.fixture
def fake_server():
    servers = []

    def start(kind, **kw):
        s = FakeServer(kind, **kw).__enter__()
        servers.append(s)
        return s

    yield start
    for s in servers:
        s.__exit__(None, None, None)


@pytest.fixture
def fake_smi_cmd():
    # forward slashes so the same string splits the same way on every OS
    return f'"{Path(sys.executable).as_posix()}" "{FAKE_SMI.as_posix()}"'
