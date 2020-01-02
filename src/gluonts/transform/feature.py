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

from typing import List

import numpy as np
import pandas as pd

from gluonts.core.component import validated, DType
from gluonts.dataset.common import DataEntry
from gluonts.time_feature import TimeFeature

from ._base import SimpleTransformation, MapTransformation
from .split import shift_timestamp


def target_transformation_length(
    target: np.array, pred_length: int, is_train: bool
) -> int:
    return target.shape[-1] + (0 if is_train else pred_length)


class AddObservedValuesIndicator(SimpleTransformation):
    """
    Replaces missing values in a numpy array (NaNs) with a dummy value and adds
    an "observed"-indicator that is ``1`` when values are observed and ``0``
    when values are missing.


    Parameters
    ----------
    target_field
        Field for which missing values will be replaced
    output_field
        Field name to use for the indicator
    dummy_value
        Value to use for replacing missing values.
    convert_nans
        If set to true (default) missing values will be replaced. Otherwise
        they will not be replaced. In any case the indicator is included in the
        result.
    """

    @validated()
    def __init__(
        self,
        target_field: str,
        output_field: str,
        dummy_value: int = 0,
        convert_nans: bool = True,
        dtype: DType = np.float32,
    ) -> None:
        self.dummy_value = dummy_value
        self.target_field = target_field
        self.output_field = output_field
        self.convert_nans = convert_nans
        self.dtype = dtype

    def transform(self, data: DataEntry) -> DataEntry:
        value = data[self.target_field]
        nan_indices = np.where(np.isnan(value))
        nan_entries = np.isnan(value)

        if self.convert_nans:
            value[nan_indices] = self.dummy_value

        data[self.target_field] = value
        # Invert bool array so that missing values are zeros and store as float
        data[self.output_field] = np.invert(nan_entries).astype(self.dtype)
        return data


class AddConstFeature(MapTransformation):
    """
    Expands a `const` value along the time axis as a dynamic feature, where
    the T-dimension is defined as the sum of the `pred_length` parameter and
    the length of a time series specified by the `target_field`.

    If `is_train=True` the feature matrix has the same length as the `target` field.
    If `is_train=False` the feature matrix has length len(target) + pred_length

    Parameters
    ----------
    output_field
        Field name for output.
    target_field
        Field containing the target array. The length of this array will be used.
    pred_length
        Prediction length (this is necessary since
        features have to be available in the future)
    const
        Constant value to use.
    dtype
        Numpy dtype to use for resulting array.
    """

    @validated()
    def __init__(
        self,
        output_field: str,
        target_field: str,
        pred_length: int,
        const: float = 1.0,
        dtype: DType = np.float32,
    ) -> None:
        self.pred_length = pred_length
        self.const = const
        self.dtype = dtype
        self.output_field = output_field
        self.target_field = target_field

    def map_transform(self, data: DataEntry, is_train: bool) -> DataEntry:
        length = target_transformation_length(
            data[self.target_field], self.pred_length, is_train=is_train
        )
        data[self.output_field] = self.const * np.ones(
            shape=(1, length), dtype=self.dtype
        )
        return data


class AddTimeFeatures(MapTransformation):
    """
    Adds a set of time features.

    If `is_train=True` the feature matrix has the same length as the `target` field.
    If `is_train=False` the feature matrix has length len(target) + pred_length

    Parameters
    ----------
    start_field
        Field with the start time stamp of the time series
    target_field
        Field with the array containing the time series values
    output_field
        Field name for result.
    time_features
        list of time features to use.
    pred_length
        Prediction length
    """

    @validated()
    def __init__(
        self,
        start_field: str,
        target_field: str,
        output_field: str,
        time_features: List[TimeFeature],
        pred_length: int,
    ) -> None:
        self.date_features = time_features
        self.pred_length = pred_length
        self.start_field = start_field
        self.target_field = target_field
        self.output_field = output_field
        self._min_time_point: pd.Timestamp = None
        self._max_time_point: pd.Timestamp = None
        self._full_range_date_features: np.ndarray = None
        self._date_index: pd.DatetimeIndex = None

    def _update_cache(self, start: pd.Timestamp, length: int) -> None:
#         end = shift_timestamp(start, length)
#         if self._min_time_point is not None:
#             if self._min_time_point <= start and end <= self._max_time_point:
#                 return
        
        #if self._min_time_point is None:
        self._min_time_point = start
            #self._max_time_point = end
#         self._min_time_point = min(
#             shift_timestamp(start, -50), self._min_time_point
#         )
#         self._max_time_point = max(
#             shift_timestamp(end, 50), self._max_time_point
#         )
        
#         self.full_date_range = pd.date_range(
#             self._min_time_point, self._max_time_point, freq=start.freq
#         )
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
        end = self._min_time_point +(length/7-1)* pd.offsets.CustomBusinessDay(holidays=holidays_NYSE) +pd.offsets.Hour(6)
        self.full_date_range = pd.date_range(
            self._min_time_point, end=end, freq=pd.offsets.CustomBusinessHour(end='16:00',holidays=holidays_NYSE)
        )#[:length]
    
        self._full_range_date_features = (
            np.vstack(
                [feat(self.full_date_range) for feat in self.date_features]
            )
            if self.date_features
            else None
        )
        self._date_index = pd.Series(
            index=self.full_date_range,
            data=np.arange(len(self.full_date_range)), #abc
        )

    def map_transform(self, data: DataEntry, is_train: bool) -> DataEntry:
        start = data[self.start_field]
        length = target_transformation_length(
            data[self.target_field], self.pred_length, is_train=is_train
        )
        self._update_cache(start, length)
        i0 = self._date_index[start]#a
        features = (
            self._full_range_date_features[..., i0 : i0 + length]
            if self.date_features
            else None
        )
        data[self.output_field] = features
        return data


class AddAgeFeature(MapTransformation):
    """
    Adds an 'age' feature to the data_entry.

    The age feature starts with a small value at the start of the time series
    and grows over time.

    If `is_train=True` the age feature has the same length as the `target`
    field.
    If `is_train=False` the age feature has length len(target) + pred_length

    Parameters
    ----------
    target_field
        Field with target values (array) of time series
    output_field
        Field name to use for the output.
    pred_length
        Prediction length
    log_scale
        If set to true the age feature grows logarithmically otherwise linearly
        over time.
    """

    @validated()
    def __init__(
        self,
        target_field: str,
        output_field: str,
        pred_length: int,
        log_scale: bool = True,
        dtype: DType = np.float32,
    ) -> None:
        self.pred_length = pred_length
        self.target_field = target_field
        self.feature_name = output_field
        self.log_scale = log_scale
        self._age_feature = np.zeros(0)
        self.dtype = dtype

    def map_transform(self, data: DataEntry, is_train: bool) -> DataEntry:
        length = target_transformation_length(
            data[self.target_field], self.pred_length, is_train=is_train
        )

        if self.log_scale:
            age = np.log10(2.0 + np.arange(length, dtype=self.dtype))
        else:
            age = np.arange(length, dtype=self.dtype)

        data[self.feature_name] = age.reshape((1, length))

        return data
