def partition(array, start, end, compare_func):
    pivot = array[start]
    low = start + 1
    high = end

    while True:
        while low <= high and compare_func(array[high], pivot):
            high = high - 1

        while low <= high and not compare_func(array[low], pivot):
            low = low + 1

        if low <= high:
            array[low], array[high] = array[high], array[low]
        else:
            break

    array[start], array[high] = array[high], array[start]

    return high


def quick_sort(array, start, end, compare_func):
    if start >= end:
        return

    p = partition(array, start, end, compare_func)

    quick_sort(array, start, p - 1, compare_func)
    quick_sort(array, p + 1, end, compare_func)
