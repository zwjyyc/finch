def bubble_sort(arr):
    n = len(arr)
    temp = None
    for i in range(n):
        for j in range(1, n-i):
            if arr[j-1] > arr[j]:
                temp = arr[j-1]
                arr[j-1] = arr[j]
                arr[j] = temp


if __name__ == '__main__':
    arr = [1, 50, 30, 10, 60, 80]
    print(arr)
    bubble_sort(arr)
    print(arr)
