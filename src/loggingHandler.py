# -*- coding: utf-8 -*-

# For longer texts or text not related to a specific progress bar
# tqdm offers write() which logs text safely to the console. To be
# precise, it adds the text above the progress bars, so each line
# you ever logged using write() will be visible.

import logging
import tqdm


class LoggingHandler(logging.Handler):
    # default: logging.info
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        # must be implemented in the logginghandler subclass
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)
