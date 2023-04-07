
def get_stats(values):
    if len(values) == 0:
        return 0, 0, 0
    values.sort()

    values_length = len(values)
    half_index = int(len(values) / 2)
    if values_length % 2 == 1:
        median = (values[half_index] + values[half_index + 1]) / 2
    else:
        median = values[half_index]

    mean = sum(values) / values_length

    protected_mean = sum(values[1:(len(values)-1)]) / (values_length - 2)

    mode = max(values)

    mean_deviation = 0
    for value in values:
        mean_deviation += abs(mean - value)

    mean_deviation = mean_deviation / values_length

    return mean, median, mean_deviation, mode, protected_mean
