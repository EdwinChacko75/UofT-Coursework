#include <stdlib.h>
#include <stdio.h>

typedef struct CircularQueue {
    int *data;
    int size;
    int capacity;
    int front;
    int rear;
} CircularQueue;

void init(CircularQueue *queue, int capacity) {
    queue->data = (int*) malloc(sizeof(int) * capacity);
    queue->size = 0;
    queue->capacity = capacity;
    queue->front = 0;
    queue->rear = -1;
}

void enqueue(CircularQueue *queue, int item) {//fix
    if (queue->size == queue->capacity) {
        int new_capacity = queue->capacity * 2;
        int* new_data = (int*) malloc(sizeof(int) * new_capacity);
        int i = 0;
        while (i < queue->size) {
            new_data[i] = queue->data[(queue->front + i) % queue->capacity];
            i++;
        }
        queue->capacity = new_capacity;
        free(queue->data);
        queue->data = new_data;
        queue->front = 0;
        queue->rear = queue->size - 1;
    }
    queue->rear = (queue->rear + 1) % queue->capacity;
    queue->data[queue->rear] = item;
    queue->size++;
}

int dequeue(CircularQueue *queue) {
    if (queue->size == 0) {
        printf("error\n");
        return -1;
    }

    int item = queue->data[queue->front];
    queue->front = (queue->front + 1) % queue->capacity;
    queue->size--;

    return item;
}

void printQueue(CircularQueue *queue) {
    printf("Queue contents: ");
    int count = 0;
    int i = queue->front;
    while (count < queue->size) {
        printf("%d ", queue->data[i]);
        i = (i + 1) % queue->capacity;
        count++;
    }
    printf("\n");
}

void freeQueue(CircularQueue *queue) {
    free(queue->data);
}
int main()
{
    CircularQueue *list = (CircularQueue*) malloc(sizeof(CircularQueue));
    init(list, 5);
    enqueue(list, 1);
    enqueue(list, 2);
    enqueue(list, 3);
    enqueue(list, 4);
    enqueue(list, 5);
    dequeue(list);
    enqueue(list, 1);
    enqueue(list, 6);
    enqueue(list, 7);
    enqueue(list, 8);

    for (int i = 0; i < list->size; i++) {
        printf("%d\n", list->data[(list->front + i) % list->capacity]);
    }

}
void enqueue(CircularQueue *q,int val){
    if(q->front ==(q->rear+1)%q->capacity){
        q->data = (int *)realloc(q->data, sizeof(int)*q->capacity*2);
        if(q->front > q->rear){
            memmove(q->data+q->capacity, q->data, sizeof(int)*q->rear);
            q->rear += q->capacity;
        } 
        q->capacity *=2;
    }
    q->data[q->rear]=val;
    q->rear = (q->rear +1)%q->capacity;

}

int dequeue(CircularQueue *q){
    int index = q->front;
    q->front = (q->front +1)%q->capacity;
    q->size--;
    return q->data[index];
}
typedef struct CircularQueue {
    int *data;
    int capacity;
    int front;
    int rear;
} CircularQueue;

void enqueue(CircularQueue *q, int val){
    if(q->front == (q->rear+1)%q->capacity){
        q->data = (int*)realloc(q->data, sizeof(int)*2*q->capacity);
        if(q->rear >q->front){
            memmove(q->data + q->capacity,q->data,q->capacity);
            q->rear += q->capacity
        }
        q->capacity*=2;
    }
    q->data[q->rear] = val;
    q->rear = (q->rear+1)%q->capacity;
}
int dequeue(CircularQueue *q){
    int index = q->front;
    q->front = (q->front+1)%q->capacity;
    return q->data[index];
}