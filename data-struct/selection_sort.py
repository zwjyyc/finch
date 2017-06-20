def selection_sort(arr):
    for i in range(len(arr)-1):
        min_idx = i
        for j in range(i+1, len(arr)):
            if arr[j] < arr[i]:
                min_idx = j
        if min_idx != i:
            temp = arr[min_idx]
            arr[min_idx] = arr[i]
            arr[i] = temp


if __name__ == '__main__':
    arr = [1, 50, 30, 10, 60, 80]
    print(arr)
    selection_sort(arr)
    print(arr)
