# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time
from collections import OrderedDict


class Meter(object):
    def __init__(self):
        pass

    def reset(self):
        raise NotImplementedError

    def update(self):
        raise NotImplementedError


class MeterBundle(object):
    def __init__(self):
        self.meters = OrderedDict()

    def reset(self):
        for meter in self.meters:
            meter.reset()

    def update(self, name, value):
        if name not in self.meters:
            meter_class = self._infer_meter_class(value)
            self.meters[name] = meter_class()

        self.meters[name].update(value)

    def write_log(self, log_writer):
        for meter in self.meters:
            meter.write_log(log_writer)

    def _infer_meter_class(self, value):
        raise NotImplementedError


class AverageMeter(Meter):
    """Computes and stores the average and current value"""

    def __init__(self):
        super(AverageMeter, self).__init__()
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class TimeMeter(object):
    """Computes the average occurrence of some event per second"""

    def __init__(self, init=0):
        self.reset(init)

    def reset(self, init=0):
        self.init = init
        self.start = time.time()
        self.n = 0

    def update(self, val=1):
        self.n += val

    @property
    def avg(self):
        return self.n / self.elapsed_time

    @property
    def elapsed_time(self):
        return self.init + (time.time() - self.start)


class StopwatchMeter(object):
    """Computes the sum/avg duration of some event in seconds"""

    def __init__(self):
        self.reset()

    def start(self):
        self.start_time = time.time()

    def stop(self, n=1):
        if self.start_time is not None:
            delta = time.time() - self.start_time
            self.sum += delta
            self.n += n
            self.start_time = None

    def reset(self):
        self.sum = 0
        self.n = 0
        self.start_time = None

    @property
    def avg(self):
        return self.sum / self.n
