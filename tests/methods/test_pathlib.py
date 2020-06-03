from __future__ import annotations

from pathlib import Path
from string import ascii_lowercase
from tempfile import TemporaryDirectory
from typing import List

from hypothesis.strategies import lists
from hypothesis.strategies import text

from tests import given
from tests import parametrize
from tests.strategies import Case
from tests.strategies import CASES


@given(x=lists(text(alphabet=ascii_lowercase, min_size=1)))
@parametrize("case", CASES)
@parametrize("use_path", [True, False])
def test_iterdir(x: List[str], use_path: bool, case: Case) -> None:
    with TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        for i in x:
            temp_dir.joinpath(i).touch()
        if use_path:
            y = case.cls.iterdir(temp_dir)
        else:
            y = case.cls.iterdir(temp_dir_str)
        assert isinstance(y, case.cls)
        assert set(y) == {temp_dir.joinpath(i) for i in case.cast(x)}
