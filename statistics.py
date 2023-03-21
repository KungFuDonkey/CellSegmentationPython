
def get_stats(values):
    if len(values) == 0:
        return 0, 0, 0
    values.sort()

    values_length = len(values)
    half_index = int(len(values) / 2)
    if values_length % 2 == 1:
        mean = (values[half_index] + values[half_index + 1]) / 2
    else:
        mean = values[half_index]

    mean_deviation = 0
    for average in values:
        mean_deviation += abs(mean - average)

    mean_deviation = mean_deviation / values_length

    average = sum(values) / values_length

    return average, mean, mean_deviation
