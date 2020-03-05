# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time
from collections import OrderedDict
import torch
import numpy


class Meter(object):
    def __init__(self):
        pass

    def reset(self):
        raise NotImplementedError

    def update(self):
        raise NotImplementedError

    def write_log(self, log_writer, name, global_step=None):
        raise NotImplementedError

    def merge(self, other):
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError


class AverageMeter(Meter):
    """Computes and stores the average and current value"""

    def __init__(self):
        super(AverageMeter, self).__init__()
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        assert n >= 0, "n cannot be negative."
        self.sum += val * n
        self.count += n

    @property
    def avg(self):
        return self.sum / self.count

    def write_log(self, log_writer, name, global_step=None):
        log_writer.add_scalar(name, self.avg, global_step)

    def merge(self, other):
        assert isinstance(other, AverageMeter)
        self.sum += other.sum
        self.count += other.count

    def __str__(self):
        return str(self.avg)


class HistogramMeter(Meter):
    """Computes and stores the all values for histogram visualisation"""

    def __init__(self):
        super(HistogramMeter, self).__init__()
        self.value_list = []
        self.reset()

    def reset(self):
        self.value_list = []

    def update(self, val):
        if isinstance(val, torch.Tensor) or isinstance(val, numpy.ndarray):
            self.value_list.append(val)
        else:
            raise ValueError("Value has invalid type.")

    def write_log(self, log_writer, name, global_step=None):
        if all(isinstance(val, torch.Tensor) for val in self.value_list):
            val_flat_list = [val.view(-1) for val in self.value_list]
            val_flat = torch.cat(val_flat_list, dim=0)
        elif all(isinstance(val, numpy.ndarray) for val in self.value_list):
            val_flat_list = [val.reshape((-1)) for val in self.value_list]
            val_flat = numpy.concatenate(val_flat_list, dim=0)
        else:
            raise ValueError("Value has invalid type.")

        log_writer.add_histogram(name, val_flat, global_step)

    def merge(self, other):
        assert isinstance(other, HistogramMeter)
        self.value_list += other.value_list


class TimeMeter(Meter):
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

    def write_log(self, log_writer, name, global_step=None):
        log_writer.add_scalar(name, self.avg, global_step)


class StopwatchMeter(Meter):
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

    def write_log(self, log_writer, name, global_step=None):
        log_writer.add_scalar(name, self.avg, global_step)


class MeterBundle(object):
    def __init__(self):
        self.meters = OrderedDict()

    def reset(self):
        for meter in self.meters.values():
            meter.reset()

    def update(self, name, value, meter_class=None):
        if name not in self.meters:
            if meter_class is None:
                meter_class = self._infer_meter_class(value)
            self.meters[name] = meter_class()

        self.meters[name].update(value)

    def write_log(self, log_writer, global_step=None):
        for name, meter in self.meters.items():
            meter.write_log(log_writer, name, global_step)

    def _infer_meter_class(self, value):
        # AverageMeter is the default meter class
        if isinstance(value, torch.Tensor):
            return HistogramMeter if value.dim() > 0 else AverageMeter

        if isinstance(value, numpy.ndarray):
            return HistogramMeter if value.ndim > 0 else AverageMeter

        if type(value) in [int, float]:
            return AverageMeter

        raise ValueError("Cannot infer meter class type for %r" % value)

    def merge(self, other):
        assert isinstance(other, MeterBundle)
        for name, meter in other.meters.items():
            if name not in self.meters:
                self.meters[name] = meter
            else:
                self.meters[name].merge(meter)

    def __iadd__(self, other):
        self.merge(other)
        return self

    def __str__(self):
        meter_values = []
        for name, meter in self.meters.items():
            try:
                value_str = str(meter)
            except:
                continue
            meter_values.append("%s=%s" % (name, value_str))

        return ", ".join(meter_values)
