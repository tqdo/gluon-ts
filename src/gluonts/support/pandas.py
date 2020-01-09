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


import pandas as pd
from gluonts.time_feature.holidaysNYSE import holidays_NYSE

def frequency_add(ts: pd.Timestamp, amount: int) -> pd.Timestamp:
    
    #return ts + ts.freq * amount
    return ts+amount*pd.offsets.CustomBusinessHour(end='16:00',holidays=holidays_NYSE)


def forecast_start(entry):
    return frequency_add(entry["start"], len(entry["target"]))
