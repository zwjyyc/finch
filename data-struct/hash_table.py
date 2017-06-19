class Entry:
    def __init__(self, key=None, val=None):
        self.key = key
        self.val = val
        self.next = None

    def get_key(self):
        return self.key

    def get_val(self):
        return self.val

class HashTable:
    def __init__(self, size):
        self.size = size
        self.hash_array = [Entry() for _ in range(self.size)]

    def get_hash(self, key):
        return key % self.size

    def put(self, key, val):
        hash_index = self.get_hash(key)
        hashed = self.hash_array[hash_index]
        new_item = Entry(key, val)
        new_item.next = hashed
        hashed.next = new_item

    def get(self, key):
        val = None
        hash_index = self.get_hash(key)
        hashed = self.hash_array[hash_index]
        while hashed.next is not None:
            if hashed.get_key() == key:
                val = hashed.get_val()
                break
            hashed = hashed.next
        return val

if __name__ == '__main__':
    ht = HashTable(10)
    ht.put(11, 'pd')
    ht.put(12, 'ax')
    ht.put(13, 'jb')
    print(ht.get(12))
