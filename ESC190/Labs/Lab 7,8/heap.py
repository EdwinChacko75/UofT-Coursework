class Heap:
    def __init__(self):
        self.heap = [None]
        self.size = 0

    
    #getting indices
    def get_left_child_index(self, parent_index):
        return 2 * parent_index
    def get_right_child_index(self, parent_index):
        return 2 * parent_index + 1
    def get_parent_index(self, child_index):
        return child_index // 2
    
    #swapping values
    def swap(self, index1, index2):
        self.heap[index1], self.heap[index2] = self.heap[index2], self.heap[index1]

    #checking family
    def has_left_child(self, index):
        return self.get_left_child_index(index) <= self.size
    def has_right_child(self, index):
        return self.get_right_child_index(index) <= self.size
    def has_parent(self, index):
        return self.get_parent_index(index) >= 1
    
    #finding family memebers
    def left_child(self, index):
        return self.heap[self.get_left_child_index(index)]
    def right_child(self, index):
        return self.heap[self.get_right_child_index(index)]
    def parent(self, index):
        return self.heap[self.get_parent_index(index)]

    #insert
    def insert(self, value):
        self.heap.append(value)
        self.size += 1
        self.heapify_up()

    def heapify_up(self):
        index = self.size 
        while self.has_parent(index) and self.parent(index) > self.heap[index]:
            self.swap(self.get_parent_index(index), index)
            index = self.get_parent_index(index)

    def heapify_down(self):
        index = 1
        while self.has_left_child(index):
            smaller_child_index = self.get_left_child_index(index)
            if self.has_right_child(index) and self.right_child(index) < self.left_child(index):
                smaller_child_index = self.get_right_child_index(index)
            if self.heap[index] < self.heap[smaller_child_index]:
                break
            else:
                self.swap(index, smaller_child_index)
            index = smaller_child_index


    def extract_min(self):
        if self.size == 0:
            return None
        min = self.heap[1]
        self.heap[1] = self.heap[self.size]
        self.heap.pop()
        self.size -= 1
        self.heapify_down()
        return min
    

if __name__ == '__main__':
    h = Heap()
    h.insert(5)
    h.insert(3)
    h.insert(8)
    print(h.extract_min())
    print(h.extract_min())
    print(h.extract_min())