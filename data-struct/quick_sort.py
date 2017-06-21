def quick_sort(array):
    less = []
    equal = []
    greater = []

    if len(array) > 1:
        pivot = array[0]
        for x in array:
            if x < pivot:
                less.append(x)
            if x == pivot:
                equal.append(x)
            if x > pivot:
                greater.append(x)
        return quick_sort(less) + quick_sort(equal) + quick_sort(greater)
    else:  # at the end of the recursion, when you only have one element in your array, just return the array.
        return array

if __name__ == '__main__':
    print(quick_sort([1, 50, 30, 10, 60, 80]))
    