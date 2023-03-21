import random
import time
import tqdm
import sys
import statistics

NUMBER_OF_BENCHMARKS_PER_INPUT = 10
WARMUP_ROUNDS = 1000


class BenchMarker:

    # takes a function as input that must be benchmarked. If you need a setup (like for unet) consider setting this
    # to a lambda function
    def __init__(self, function, name):
        self.name = name
        self.function = function
        self.averages = []

    # Runs a full benchmark over the function, this benchmark provides inputs for the function.
    # in our case the inputs can be filled in with opencv images
    def run_full_benchmark(self, inputs):
        start_time = time.time()
        print('running benchmarker {name}'.format(name=self.name))
        print('warming up')
        for x in tqdm.tqdm(range(WARMUP_ROUNDS), file=sys.stdout):
            self.function(random.choice(inputs))

        print('performing benchmark')
        for i in tqdm.tqdm(range(len(inputs)), file=sys.stdout):
            self.run_benchmark_with_input(inputs[i])

        print(
            'finished benchmarker {name} in {time} seconds'.format(name=self.name, time=int(time.time() - start_time)))

    # Runs as benchmark over one input. It was written this way as time can differ for different inputs.
    def run_benchmark_with_input(self, function_input):
        total_time = 0
        for i in range(1, NUMBER_OF_BENCHMARKS_PER_INPUT):
            start_time = time.time()
            self.function(function_input)
            end_time = time.time()
            total_time += end_time - start_time

        avg_result = total_time / NUMBER_OF_BENCHMARKS_PER_INPUT

        # multiply by 1000 for MS instead of seconds
        self.averages += [avg_result * 1000]

    # gets the results of the benchmark. Returns (avg, mean, mean_dev) in MS
    def get_benchmark_results(self):
        return statistics.get_stats(self.averages)
