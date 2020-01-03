# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

"""
Train/test splitter
~~~~~~~~~~~~~~~~~~~

This module defines strategies to split a whole dataset into train and test
subsets.

For uniform datasets, where all time-series start and end at the same point in
time `OffsetSplitter` can be used::

    splitter = OffsetSplitter(prediction_length=24, split_offset=24)
    train, test = splitter.split(whole_dataset)

For all other datasets, the more flexible `DateSplitter` can be used::

    splitter = DateSplitter(
        prediction_length=24,
        split_date=pd.Timestamp('2018-01-31', freq='D')
    )
    train, test = splitter.split(whole_dataset)

The module also supports rolling splits::

    splitter = DateSplitter(
        prediction_length=24,
        split_date=pd.Timestamp('2018-01-31', freq='D')
    )
    train, test = splitter.rolling_split(whole_dataset, windows=7)
"""

# Standard library imports
from abc import ABC, abstractmethod
from typing import List, Optional

# Third-party imports
import pandas as pd
import pydantic

# First-party imports
from gluonts.dataset.common import TimeSeriesItem


class TimeSeriesSlice(pydantic.BaseModel):
    """Like TimeSeriesItem, but all time-related fields are of type pd.Series
    and is indexable, e.g `ts_slice['2018':]`.
    """

    class Config:
        arbitrary_types_allowed = True

    target: pd.Series
    item: str

    feat_static_cat: List[int] = []
    feat_static_real: List[float] = []

    feat_dynamic_cat: List[pd.Series] = []
    feat_dynamic_real: List[pd.Series] = []

    @classmethod
    def from_time_series_item(
        cls, item: TimeSeriesItem, freq: Optional[str] = None
    ) -> "TimeSeriesSlice":
        if freq is None:
            freq = item.start.freq
        holidays_NYSE=['1998-04-10',
         '1998-05-25',
         '1998-07-03',
         '1998-09-07',
         '1998-11-26',
         '1998-12-25',
         '1999-01-01',
         '1999-01-18',
         '1999-02-15',
         '1999-04-02',
         '1999-05-31',
         '1999-07-05',
         '1999-09-06',
         '1999-11-25',
         '1999-12-24',
         '2000-01-01',
         '2000-01-17',
         '2000-02-21',
         '2000-04-21',
         '2000-05-29',
         '2000-07-04',
         '2000-09-04',
         '2000-11-23',
         '2000-12-25',
         '2001-01-01',
         '2001-01-15',
         '2001-02-19',
         '2001-04-13',
         '2001-05-28',
         '2001-07-04',
         '2001-09-03',
         '2001-09-11',
         '2001-09-12',
         '2001-09-13',
         '2001-09-14',
         '2001-09-15',
         '2001-09-16',
         '2001-11-22',
         '2001-12-25',
         '2002-01-01',
         '2002-01-21',
         '2002-02-18',
         '2002-03-29',
         '2002-05-27',
         '2002-07-04',
         '2002-09-02',
         '2002-11-28',
         '2002-12-25',
         '2003-01-01',
         '2003-01-20',
         '2003-02-17',
         '2003-04-18',
         '2003-05-26',
         '2003-07-04',
         '2003-09-01',
         '2003-11-27',
         '2003-12-25',
         '2004-01-01',
         '2004-01-19',
         '2004-02-16',
         '2004-04-09',
         '2004-05-31',
         '2004-06-11',
         '2004-07-05',
         '2004-09-06',
         '2004-11-25',
         '2004-12-24',
         '2005-01-01',
         '2005-01-17',
         '2005-02-21',
         '2005-03-25',
         '2005-05-30',
         '2005-07-04',
         '2005-09-05',
         '2005-11-24',
         '2005-12-26',
         '2006-01-02',
         '2006-01-16',
         '2006-02-20',
         '2006-04-14',
         '2006-05-29',
         '2006-07-04',
         '2006-09-04',
         '2006-11-23',
         '2006-12-25',
         '2007-01-01',
         '2007-01-02',
         '2007-01-15',
         '2007-02-19',
         '2007-04-06',
         '2007-05-28',
         '2007-07-04',
         '2007-09-03',
         '2007-11-22',
         '2007-12-25',
         '2008-01-01',
         '2008-01-21',
         '2008-02-18',
         '2008-03-21',
         '2008-05-26',
         '2008-07-04',
         '2008-09-01',
         '2008-11-27',
         '2008-12-25',
         '2009-01-01',
         '2009-01-19',
         '2009-02-16',
         '2009-04-10',
         '2009-05-25',
         '2009-07-03',
         '2009-09-07',
         '2009-11-26',
         '2009-12-25',
         '2010-01-01',
         '2010-01-18',
         '2010-02-15',
         '2010-04-02',
         '2010-05-31',
         '2010-07-05',
         '2010-09-06',
         '2010-11-25',
         '2010-12-24',
         '2011-01-01',
         '2011-01-17',
         '2011-02-21',
         '2011-04-22',
         '2011-05-30',
         '2011-07-04',
         '2011-09-05',
         '2011-11-24',
         '2011-12-26',
         '2012-01-02',
         '2012-01-16',
         '2012-02-20',
         '2012-04-06',
         '2012-05-28',
         '2012-07-04',
         '2012-09-03',
         '2012-10-29',
         '2012-10-30',
         '2012-11-22',
         '2012-12-25',
         '2013-01-01',
         '2013-01-21',
         '2013-02-18',
         '2013-03-29',
         '2013-05-27',
         '2013-07-04',
         '2013-09-02',
         '2013-11-28',
         '2013-12-25',
         '2014-01-01',
         '2014-01-20',
         '2014-02-17',
         '2014-04-18',
         '2014-05-26',
         '2014-07-04',
         '2014-09-01',
         '2014-11-27',
         '2014-12-25',
         '2015-01-01',
         '2015-01-19',
         '2015-02-16',
         '2015-04-03',
         '2015-05-25',
         '2015-07-03',
         '2015-09-07',
         '2015-11-26',
         '2015-12-25',
         '2016-01-01',
         '2016-01-18',
         '2016-02-15',
         '2016-03-25',
         '2016-05-30',
         '2016-07-04',
         '2016-09-05',
         '2016-11-24',
         '2016-12-26',
         '2017-01-02',
         '2017-01-16',
         '2017-02-20',
         '2017-04-14',
         '2017-05-29',
         '2017-07-04',
         '2017-09-04',
         '2017-11-23',
         '2017-12-25',
         '2018-01-01',
         '2018-01-15',
         '2018-02-19',
         '2018-03-30',
         '2018-05-28',
         '2018-07-04',
         '2018-09-03',
         '2018-11-22',
         '2018-12-05',
         '2018-12-25',
         '2019-01-01',
         '2019-01-21',
         '2019-02-18',
         '2019-04-19',
         '2019-05-27',
         '2019-07-04',
         '2019-09-02',
         '2019-11-28',
         '2019-12-25',
         '2020-01-01',
         '2020-01-20',
         '2020-02-17',
         '2020-04-10',
         '2020-05-25',
         '2020-07-03',
         '2020-09-07',
         '2020-11-26',
         '2020-12-25']
        
        index = pd.date_range(
            start=item.start, freq=pd.offsets.CustomBusinessHour(end='16:00',holidays=holidays_NYSE), end=item.start+(len(item.target)/7-1)*pd.offsets.CustomBusinessDay(holidays=holidays_NYSE)+pd.offsets.Hour(6)
        )

        feat_dynamic_cat = [
            pd.Series(cat, index=index) for cat in item.feat_dynamic_cat
        ]

        feat_dynamic_real = [
            pd.Series(real, index=index) for real in item.feat_dynamic_real
        ]

        return TimeSeriesSlice(
            target=pd.Series(item.target, index=index),
            item=item.item,
            feat_static_cat=item.feat_static_cat,
            feat_static_real=item.feat_static_real,
            feat_dynamic_cat=feat_dynamic_cat,
            feat_dynamic_real=feat_dynamic_real,
        )

    def to_time_series_item(self) -> TimeSeriesItem:
        return TimeSeriesItem(
            start=self.start,
            target=self.target.values,
            item=self.item,
            feat_static_cat=self.feat_static_cat,
            feat_static_real=self.feat_static_real,
            feat_dynamic_cat=[cat.values for cat in self.feat_dynamic_cat],
            feat_dynamic_real=[real.values for real in self.feat_dynamic_real],
        )

    @property
    def start(self):
        return self.target.index[0]

    @property
    def end(self):
        return self.target.index[-1]

    def __len__(self) -> int:
        return len(self.target)

    def __getitem__(self, slice_: slice) -> "TimeSeriesSlice":
        feat_dynamic_real = None
        feat_dynamic_cat = None

        if self.feat_dynamic_real is not None:
            feat_dynamic_real = [feat for feat in self.feat_dynamic_real]

        if self.feat_dynamic_cat is not None:
            feat_dynamic_cat = [feat for feat in self.feat_dynamic_cat]

        return TimeSeriesSlice(
            item=self.item,
            target=self.target[slice_],
            feat_dynamic_cat=feat_dynamic_cat,
            feat_dynamic_real=feat_dynamic_real,
            feat_static_cat=self.feat_static_cat,
            feat_static_real=self.feat_static_real,
        )


class TrainTestSplit(pydantic.BaseModel):
    train: List[TimeSeriesItem] = []
    test: List[TimeSeriesItem] = []

    def _add_train_slice(self, train_slice: TimeSeriesSlice) -> None:
        # is there any data left for training?
        if train_slice:
            self.train.append(train_slice.to_time_series_item())

    def _add_test_slice(self, test_slice: TimeSeriesSlice) -> None:
        self.test.append(test_slice.to_time_series_item())


class AbstractBaseSplitter(ABC):
    """Base class for all other splitter.

    Args:
        param prediction_length:
            The prediction length which is used to train themodel.

        max_history:
            If given, all entries in the *test*-set have a max-length of
            `max_history`. This can be sued to produce smaller file-sizes.
    """

    # @property
    # @abstractmethod
    # def prediction_length(self) -> int:
    #     pass

    # @property
    # @abstractmethod
    # def max_history(self) -> Optional[int]:
    #     pass

    @abstractmethod
    def _train_slice(self, item: TimeSeriesSlice) -> TimeSeriesSlice:
        pass

    @abstractmethod
    def _test_slice(
        self, item: TimeSeriesSlice, offset: int = 0
    ) -> TimeSeriesSlice:
        pass

    def _trim_history(self, item: TimeSeriesSlice) -> TimeSeriesSlice:
        if getattr(self, "max_history") is not None:
            return item[: -getattr(self, "max_history")]
        else:
            return item

    def split(self, items: List[TimeSeriesItem]) -> TrainTestSplit:
        split = TrainTestSplit()

        for item in map(TimeSeriesSlice.from_time_series_item, items):
            train = self._train_slice(item)
            test = self._trim_history(self._test_slice(item))

            split._add_train_slice(train)

            assert len(test) - len(train) >= getattr(self, "prediction_length")
            split._add_test_slice(test)

        return split

    def rolling_split(
        self,
        items: List[TimeSeriesItem],
        windows: int,
        distance: Optional[int] = None,
    ) -> TrainTestSplit:
        # distance defaults to prediction_length
        if distance is None:
            distance = getattr(self, "prediction_length")
        assert distance is not None

        split = TrainTestSplit()

        for item in map(TimeSeriesSlice.from_time_series_item, items):
            train = self._train_slice(item)
            split._add_train_slice(train)

            for window in range(windows):
                offset = window * distance
                test = self._trim_history(
                    self._test_slice(item, offset=offset)
                )

                assert len(test) - len(train) >= getattr(self, "max_history")
                split._add_test_slice(test)

        return split


class OffsetSplitter(pydantic.BaseModel, AbstractBaseSplitter):
    "Requires uniform data."

    prediction_length: int
    split_offset: int
    max_history: Optional[int] = None

    def _train_slice(self, item: TimeSeriesSlice) -> TimeSeriesSlice:
        return item[: self.split_offset]

    def _test_slice(
        self, item: TimeSeriesSlice, offset: int = 0
    ) -> TimeSeriesSlice:
        offset_ = self.split_offset + offset + self.prediction_length
        assert offset_ <= len(item)
        return item[:offset_]


class DateSplitter(AbstractBaseSplitter, pydantic.BaseModel):
    prediction_length: int
    split_date: pd.Timestamp
    max_history: Optional[int] = None

    def _train_slice(self, item: TimeSeriesSlice) -> TimeSeriesSlice:
        # the train-slice includes everything up to (including) the split date
        return item[: self.split_date]

    def _test_slice(
        self, item: TimeSeriesSlice, offset: int = 0
    ) -> TimeSeriesSlice:
        return item[: self.split_date + self.prediction_length + offset]
