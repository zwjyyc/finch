class Node:
    def __init__(self, key, val, next):
        self.key = key
        self.val = val
        self.next = next


class LinkedList:
    def __init__(self):
        self.head = None

    def add(self, key, val):
        new_node = Node(key, val, None)
        if self.head is None:
            self.head = new_node
        else:
            new_node.next = self.head
            self.head = new_node


class HashTable:
    def __init__(self, size):
        self.size = size
        self.hash_array = [LinkedList() for _ in range(self.size)]

    def get_hash(self, key):
        return key % self.size

    def put(self, key, val):
        hash_index = self.get_hash(key)
        linked_list = self.hash_array[hash_index]
        new_item = linked_list.add(key, val)

    def get(self, key):
        val = None
        hash_index = self.get_hash(key)
        node = self.hash_array[hash_index].head
        while node is not None:
            if node.key == key:
                val = node.val 
                break
            node = node.next
        return val


if __name__ == '__main__':
    ht = HashTable(10)
    ht.put(11, 'pd')
    ht.put(12, 'ax')
    ht.put(13, 'jb')
    ht.put(21, 'ab')
    print(ht.get(21))
