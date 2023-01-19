"""
(c) 2022 Twente Medical Systems International B.V., Oldenzaal The Netherlands

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

#######  #     #   #####   #
   #     ##   ##  #        
   #     # # # #  #        #
   #     #  #  #   #####   #
   #     #     #        #  #
   #     #     #        #  #
   #     #     #  #####    #

/**
 * @file decorators.py 
 * @brief 
 * Decorator class to handle decorative functions.
 */


"""

import time
import os

from .tmsi_logger import TMSiLoggerPerformance


def LogPerformances(func):
    def performance_logger(*args, **kwargs):
        if "TMSi_ENV" in os.environ:
            env = os.environ["TMSi_ENV"]
        else:
            env = "DLL"
        if "TMSi_PERF" not in os.environ:
            response = func(*args, **kwargs)
        elif os.environ["TMSi_PERF"] == "ON":
            tic = time.perf_counter()
            response = func(*args, **kwargs)
            toc = time.perf_counter()
            TMSiLoggerPerformance().log(
                "{} | {}: {:.3f} ms".format(
                    env, func.__qualname__, (toc - tic) * 1_000
                )
            )
        else:
            response = func(*args, **kwargs)
        return response

    return performance_logger
