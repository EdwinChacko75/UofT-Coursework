#include <stdio.h>
#include <stdlib.h>
#include <string.h>


int mystrlen(char *str){
    int i = 0;
    while( str[i]!= '\0'){
        i++;
    }
    return i;
}
void f(int *p){
    *p=10;
}
int get_list_length(int *list) {
    int length = 0;
    while (*list != NULL) {
        length++;
        list++;
    }
    return length;
}
void g(char **arr){
    *arr = "Hello World!";
}
void insertion(int *arr){
    int temp;
    int len = get_list_length(arr) -1;
    for(int i = 0;i<len;i++){
        for(int j=0;j<len ;j++){
            if(arr[j]>arr[j+1]){
                temp = arr[j];
                arr[j] = arr[j+1];
                arr[j+1] = temp;
            }
        }
    }
    int i, j, key;
    for (i = 1; i < len; i++){
        key = arr[i];
        j = i - 1;
        while (j >= 0 && arr[j] > key){
            arr[j+1] = arr[j];
            j = j - 1;
        }
        arr[j+1] = key;
    }
}
void seq_replace(int *arr1, unsigned long arr1_sz, int *arr2, unsigned long arr2_sz){
    int c = 0;
    for(int i =0; i<arr1_sz;i++){
        int temp = 0;
        int j = 0;
        for(j;j < arr2_sz;j++){
            if (arr1[j] ==arr2[j]){
                temp++;
            }
        }
        if (temp == j){
            for(int k = i;k < i+arr2_sz;k++){
                arr1[k] =0;
            }
        }
        
    }
    
}
int main(){
    int a = 5 ;
    printf("Before: %d\n", a);
    f(&a);
    printf("After: %d\n",a);

    char *arr = "hello world";
    printf("Before: %s\n",arr);
    g(&arr);
    printf("After: %s\n",arr);


    int array[] = {5,1,3,2,4};
    insertion(array);
    for(int i=0;i<sizeof(array)/ sizeof(int);i++){
        printf("%d ", array[i]);
    }
    printf("\nLength of arr: %d\n", mystrlen(arr));

    int aa[] = {5, 6, 7, 8, 6, 7};
    int b[] = {6, 7};
    seq_replace(aa, 6, b, 2);
    for(int i=0;i<6;i++){
        printf("%d ", aa[i]);
    }
    
}