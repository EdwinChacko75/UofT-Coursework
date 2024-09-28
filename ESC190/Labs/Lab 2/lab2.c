#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
void getStr(int size, char *word){
    printf("How many chars is ur word: ");
    scanf("%d", &size);
    printf("What is your word: ");
    scanf("%s", &word);
    printf("%s",&word);

}
int my_strcmp_rec(char s1[], char s2[]){
    int i =0;
    if (s1[i] >s2[i]){
        return 1;
    } else if (s1[i] < s2[i]){
        return -1;
    }
    else{
        i++;
        return my_strcmp_rec(s1, s2);
    }

}

char * strcatIndex(char *arr1, char *arr2)
{
   int i, j;
   for (i = 0; arr1[i] != '\0'; i++)
   {

    }
   for (j = 0; arr2[j] != '\0'; j++)
   {
      arr1[i + j] = arr2[j];
   }
   arr1[i + j] = '\0';
   return arr1;
}
char *strcatPtr(char *arr1, char *arr2){
    int i, j;
    for (i = 0; *(arr1 + i) != '\0'; i++)
    {
    }
    for (j=0; *(arr2 + j) != '\0'; j++){
        *(arr1 + i + j) = *(arr2 + j);
    }
    *(arr1 + i + j) = '\0';

    return arr1;
}



int my_atoi(char *str){
    int j = 0;
    int nums[]={};//use double ptrs
    for(int i = 0; *(str + i) != '\0';i++){

        if (isdigit(*(str + i)) > 0){
            nums = (*(str + i)) - 0;//broken line
            j++;
        }
    }
    return *nums;
}
int main(){
    int size;
    char **str = malloc(sizeof(char *) * 5);
    //getStr(size, *str);
    //printf("%s",*str);
    free(str);
    char hello[] = "Hello ";
    char world[] = "World";
    /*
    s1 == s2 checks if the string have same value and returns a boolean
    *s1 == *s2 checks if the adresses are equal. might be different based on definition of s1,s2
    strcmp(s1, s2) returns 0 if strings r equal, positive number if str1 > s2
    and negative num if str2>str1
    */

    char s1[]= "hello";
    char s2[] = "Hello";
    //printf("Original: %d\nMy: %d",strcmp(s1,s2),my_strcmp_rec(s1,s2));
    //printf("%s",strcatIndex(hello,world));
    //printf("%s",strcatPtr(hello,world));
    char string[] = "24";
    printf("%d", my_atoi(*str));


}