from functional_itertools.utilities import Sentinel
from functional_itertools.utilities import sentinel
from hypothesis.strategies import data
from hypothesis.strategies import DataObject
from hypothesis.strategies import just
from hypothesis.strategies import register_type_strategy


register_type_strategy(DataObject, data())
register_type_strategy(Sentinel, just(sentinel))
