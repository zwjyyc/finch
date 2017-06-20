def binary_search(searched, arr):
    low = 0
    high = len(arr) - 1
    while (low <= high):
        mid = int(low + (high - low) / 2)
        if searched < arr[mid]:
            high = mid - 1
        elif searched > arr[mid]:
            low = mid + 1
        else:
            print("Found")
            return mid
    print("Not Found")
    return None


if __name__ == '__main__':
    binary_search(6, [0, 1, 5, 6, 7, 8, 11, 13])
