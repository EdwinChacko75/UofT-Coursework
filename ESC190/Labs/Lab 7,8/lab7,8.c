typedef struct heap{
    int *arr;
    int size;
    int capacity;
} heap;

void create_heap(heap *h, int capacity){
    h->arr = (int *)malloc(sizeof(int) * capacity);
    h->size = 0;
    h->capacity = capacity;
}

void swap(int *a, int *b){
    int temp = *a;
    *a = *b;
    *b = temp;
}

void insert(heap *h, int val){
    if(h->size == h->capacity){
        h->arr = (int *)realloc(h->arr, sizeof(int) * h->capacity * 2);
        h->capacity *= 2;
    }
    h->arr[h->size+1] = val;
    h->size++;
    int i = h->size;
    while(i > 1 && h->arr[i] < h->arr[i / 2]){
        swap(&h->arr[i], &h->arr[i / 2]);
        i = i / 2;
    }
}

int extract_min(heap *h)
{
    int min = h->arr[1];
    h->arr[1] = h->arr[h->size];
    h->size--;
    int i = 1;
    while(i <= h->size){
        int left = 2 * i;
        int right = 2 * i + 1;
        if(left > h->size){
            break;
        }
        if(right > h->size){
            if(h->arr[i] > h->arr[left]){
                swap(&h->arr[i], &h->arr[left]);
            }
            break;
        }
        if(h->arr[i] > h->arr[left] || h->arr[i] > h->arr[right]){
            if(h->arr[left] < h->arr[right]){
                swap(&h->arr[i], &h->arr[left]);
                i = left;
            }
            else{
                swap(&h->arr[i], &h->arr[right]);
                i = right;
            }
        }
        else{
            break;
        }
    }
    return min;
}
