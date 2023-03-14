import timeit
import time
import tqdm

NUMBER_OF_BENCHMARKS = 100


class BenchMarker:

    # takes a function as input that must be benchmarked. If you need a setup (like for unet) consider setting this
    # to a lambda function
    def __init__(self, function, name, finalizer_function=None):
        self.name = name
        self.function = function
        self.averages = []
        self.finalizer_function = finalizer_function
        self.function_result = None

    # Runs a full benchmark over the function, this benchmark provides inputs for the function.
    # in our case the inputs can be filled in with opencv images
    def run_full_benchmark(self, inputs):
        print('running benchmarker {name}'.format(name=self.name))

        # tqdm is for a progressbar in the console, iterates over inputs
        for i in tqdm.tqdm(range(len(inputs))):
            self.run_benchmark_with_input(inputs[i])

            # call finalizer function if available
            if self.finalizer_function is not None:
                self.finalizer_function(self.function_result, i)

    # runs function and stores result in function_result
    def run_function(self, function_input):
        self.function_result = self.function(function_input)

    # Runs as benchmark over one input. It was written this way as time can differ for different inputs.
    def run_benchmark_with_input(self, function_input):
        avg_result = timeit.Timer(lambda: self.run_function(function_input)).timeit(NUMBER_OF_BENCHMARKS)
        self.averages += [avg_result]

    # gets the results of the benchmark. Returns (avg, mean, mean_dev)
    def get_benchmark_results(self):
        if len(self.averages) == 0:
            return 0, 0, 0
        self.averages.sort()

        averages_length = len(self.averages)
        half_index = int(len(self.averages) / 2)
        if averages_length % 2 == 1:
            mean_completion_time = (self.averages[half_index] + self.averages[half_index + 1]) / 2
        else:
            mean_completion_time = self.averages[half_index]

        mean_deviation = 0
        for average in self.averages:
            mean_deviation += abs(mean_completion_time - average)

        mean_deviation = mean_deviation / averages_length

        average_completion_time = sum(self.averages) / len(self.averages)

        return average_completion_time, mean_completion_time, mean_deviation
