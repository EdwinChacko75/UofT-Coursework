#include <stdlib.h>
#include <stdio.h>
#include <string.h>

typedef struct node{
    int data;
    struct node *next;
} node;


typedef struct LL{
    node *head;
    int size;
} LL;


void create_node(node **p_n, int data)
{
    *p_n = (node*)malloc(sizeof(node));
    (*p_n)->next = NULL;
    (*p_n)->data = data;
}

// created a linked list that looks like data[0]->data[1]->data[2]->...->data[size-1]
void create_LL_from_data(LL **p_LL, int *data_arr, int size)
{
    (*p_LL) = (LL*)malloc(sizeof(LL));
    (*p_LL)->size = 0;
    node *cur = NULL;
    for(int i = 0; i < size; i++){
        node *n;
        create_node(&n, data_arr[i]); // n is a pointer to a node with data = data[i], next = NULL
        if(cur == NULL){
            (*p_LL)->head = n;
        }
        else{
            cur->next = n;
        }
        cur = n;
        (*p_LL)->size++;
    }
}


void LL_append(LL *my_list, int new_elem)
{
    node *cur = my_list->head;
    while(cur->next != NULL){
        cur = cur->next;
    }
    node *n;
    create_node(&n, new_elem);
    cur->next = n;
    my_list->size++;
}

void LL_insert(LL *my_list, int new_elem, int index)
{
    node *new = (node*)malloc(sizeof(LL));
    new->data = new_elem;
    node *cur = my_list->head;
    for(int i = 0; i < index-1; i++){
        cur = cur->next;
    }
    new->next = cur->next;
    cur->next = new;
    my_list->size++;
}

void LL_delete(LL *my_list, int index)
{
    node *cur = my_list->head;
    for(int i = 0; i < index-1; i++){
        cur = cur->next;
    }
    node *temp = cur->next;
    cur->next = cur->next->next;
    free(temp);

    my_list->size--;
}

void LL_free_all(LL *my_list)
{
    node *cur= my_list->head;
    for(int i = 0; i < my_list->size; i++){
        node *temp = cur;
        cur =cur->next;
        free(temp);
    }
    free(cur);

}

int helper(node *cur, int index){
    if (index == 0){
        return cur->data;
    }
    else{
        cur = cur->next;
        helper(cur, index-1);
    }
}
int LL_get_rec(LL *my_list, int index)
{
    if (index == 0){
        return my_list->head->data;
    }
    else{
        node *cur = my_list->head;
        return helper(cur, index);
    }
}



typedef struct ArrayList{
    int *data;
    int size;   // (a->data)[1]
    int capacity;
} ArrayList;

void create_AL_from_data(ArrayList **p_AL, int *data_arr, int size)
{
    (*p_AL) = (ArrayList*)malloc(sizeof(ArrayList));
    (*p_AL)->size = size;
    (*p_AL)->capacity = size;
    (*p_AL)->data = (int*) malloc(sizeof(int) * size);

    for(int i = 0; i < size; i++){
        (*p_AL)->data[i] = data_arr[i];
    }
}
void size_check(ArrayList *my_list){
    if (my_list->size == my_list->capacity){
        my_list->data = (int*) realloc(my_list->data, sizeof(int) * 2*my_list->capacity);
    }
}
void AL_append(ArrayList *my_list, int new_elem){
    size_check(my_list);
    my_list->data[my_list->size] = new_elem;
    my_list->size++;
}

void AL_insert(ArrayList *my_list, int new_elem, int index)
{
    size_check(my_list);
    for(int i = my_list->size; i>=index ;i--){
        my_list->data[i+1] = my_list->data[i];
    }
    my_list->data[index] = new_elem;
    my_list->size++;
}

void AL_delete(ArrayList *my_list, int index)
{
    for(int i = index;i<my_list->size; i++){
        my_list->data[i] = my_list->data[i+1];
    }
    my_list->size--;
}

void AL_free(ArrayList *my_list)
{
    free(my_list->data);
}


int main()
{
    int data_arr[] = {1, 2, 3, 4, 5};
    LL *my_list;
    create_LL_from_data(&my_list, data_arr, 5);
    LL_append(my_list, 6);
    LL_insert(my_list, 7, 3);
    LL_delete(my_list, 3);
    LL_free_all(my_list);
    printf("%d\n",LL_get_rec(my_list, 3));
    node *cur = my_list->head;
    while(cur != NULL){
        printf("%d\n", cur->data);
        cur = cur->next;
    }
    ArrayList *list;
    // create_AL_from_data(&list, data_arr, 5);
    // AL_append(list, 6);
    // AL_insert(list, 7, 3);
    // AL_delete(list, 3);
    // AL_free(list);
    // for(int i = 0; i < list->size; i++){
    //     printf("%d\n", list->data[i]);
    // }
    return 0;
}