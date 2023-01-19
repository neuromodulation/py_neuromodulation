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
 * @file tmsi_logger.py 
 * @brief 
 * Loggers used to handle console or file output for informative and debug reasons.
 */


"""

import os
from sys import platform
import logging
from .logger_filter import PERFORMANCE_LOG, LoggerFilter, ACTIVITY_LOG
from .singleton import Singleton
from .support_functions import get_documents_path

logging.addLevelName(PERFORMANCE_LOG, "PERFORMANCE")
logging.addLevelName(ACTIVITY_LOG, "ACTIVITY")
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")


class TMSiLogger(metaclass=Singleton):
    def __init__(self):
        self.__tmsi_log = logging.getLogger("TMSi")
        debug_stream_handler = logging.StreamHandler()
        debug_stream_handler.setFormatter(formatter)
        self.__tmsi_log.handlers = [debug_stream_handler]

    def critical(self, message):
        self.__tmsi_log.log(level=logging.CRITICAL, msg=message)

    def debug(self, message):
        self.__tmsi_log.log(level=logging.DEBUG, msg=message)

    def info(self, message):
        self.__tmsi_log.log(level=logging.INFO, msg=message)

    def warning(self, message):
        self.__tmsi_log.log(level=logging.WARNING, msg=message)


class TMSiLoggerActivity(metaclass=Singleton):
    def __init__(self):
        self.__tmsi_perf = logging.getLogger("TMSiActivity")
        perf_handler = logging.FileHandler("__activity.log")
        perf_handler.setFormatter(formatter)
        perf_handler.setLevel(ACTIVITY_LOG)
        perf_handler.addFilter(LoggerFilter(ACTIVITY_LOG))
        self.__tmsi_perf.handlers = [perf_handler]

    def log(self, message):
        self.__tmsi_perf.log(level=ACTIVITY_LOG, msg=message)


class TMSiLoggerPerformance(metaclass=Singleton):
    def __init__(self):
        self.__tmsi_perf = logging.getLogger("TMSiPerformance")
        if platform == "win32":
            tmsifolder = os.path.join(
                get_documents_path(), "TMS_International_B.V", "Apex"
            )
            if not os.path.exists(tmsifolder):
                os.makedirs(tmsifolder)
            perf_handler = logging.FileHandler(
                os.path.join(tmsifolder, "__performance.log")
            )
        else:
            perf_handler = logging.FileHandler("__performance.log")
        perf_handler.setFormatter(formatter)
        perf_handler.setLevel(PERFORMANCE_LOG)
        perf_handler.addFilter(LoggerFilter(PERFORMANCE_LOG))
        self.__tmsi_perf.handlers = [perf_handler]

    def log(self, message):
        self.__tmsi_perf.log(level=PERFORMANCE_LOG, msg=message)
