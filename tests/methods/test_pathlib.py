from __future__ import annotations

from pathlib import Path
from string import ascii_lowercase
from tempfile import TemporaryDirectory
from typing import List

from hypothesis import given
from hypothesis.strategies import lists
from hypothesis.strategies import text
from pytest import mark

from tests.strategies import Case
from tests.strategies import CASES


@mark.parametrize("case", CASES)
@mark.parametrize("use_path", [True, False])
@given(x=lists(text(alphabet=ascii_lowercase, min_size=1)))
def test_iterdir(case: Case, x: List[str], use_path: bool) -> None:
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
