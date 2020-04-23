from __future__ import annotations

from pathlib import Path
from string import ascii_lowercase
from tempfile import TemporaryDirectory
from typing import Type

from hypothesis import given
from hypothesis.strategies import data
from hypothesis.strategies import DataObject
from hypothesis.strategies import text
from pytest import mark

from tests.strategies import CLASSES
from tests.strategies import siterables


@mark.parametrize("cls", CLASSES)
@mark.parametrize("use_path", [True, False])
@given(data=data())
def test_iterdir(cls: Type, data: DataObject, use_path: bool) -> None:
    x, cast = data.draw(siterables(cls, text(alphabet=ascii_lowercase, min_size=1), unique=True))
    with TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        for i in x:
            temp_dir.joinpath(i).touch()
        if use_path:
            y = cls.iterdir(temp_dir)
        else:
            y = cls.iterdir(temp_dir_str)
        assert isinstance(y, cls)
        assert set(y) == {temp_dir.joinpath(i) for i in x}
