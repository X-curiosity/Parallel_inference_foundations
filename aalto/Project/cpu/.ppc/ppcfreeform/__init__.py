from typing import Optional
import ppcgrader.config


class Config(ppcgrader.config.Config):
    def __init__(self, code: str, gpu: bool, openmp: bool):
        from . import info
        super().__init__(binary='freeform',
                         gpu=gpu,
                         openmp=openmp,
                         cfg_file=__file__,
                         info=info,
                         code=code)
        self.export_streams = True

    def parse_output(self, output):
        input_data = {
            "x": None,
        }
        output_data = {
            "result": None,
        }
        output_errors = {}
        statistics = {}

        for line in output.splitlines():
            splitted = line.split('\t')
            if splitted[0] == 'result':
                errors = {
                    'fail': True,
                    'pass': False,
                    'done': False
                }[splitted[1]]
            elif splitted[0] == 'time':
                time = float(splitted[1])
            elif splitted[0] == 'perf_wall_clock_ns':
                time = int(splitted[1]) / 1e9
                statistics[splitted[0]] = int(splitted[1])
            elif splitted[0].startswith('perf_'):
                statistics[splitted[0]] = int(splitted[1])
            elif splitted[0] == 'input':
                input_data['x'] = int(splitted[1])
            elif splitted[0] == 'output':
                output_data["result"] = int(splitted[1])

        return time, errors, input_data, output_data, output_errors, statistics
