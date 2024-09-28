#include <stdio.h>

double atoff(const char *line){
    int i = 0;
    int sign = 1;
    double tens = 0.1;
    double res = 0;
    while(line[i] != '='){
        i++;
    }
    i++;
    sign = line[i] == '-' ? -1 : 1;
    i++;
    while(line[i] != '.'){
        res = res * 10 + (*(line + i) - '0');
        i++;
    }
    i++;
    while(line[i] != '\n'){
        res += (line[i] - '0' ) * tens;
        tens *= 0.1;
        i++;
    }
    return res * sign;
}

void readatof(char *filename){
    FILE *fp = fopen(filename,"r");
    char line[100];
    double res = 0;

    if(fp == NULL){
        printf("file dne");
        return;
    }
    while(fgets(line, 100, fp) != NULL){
        res += atoff(line);
    }
    fclose(fp);
    printf("%f", res);
}

// def change_name(s, new_name):
//      s[1] = new_name


// s = [20, "Bob"]
// change_name(s, "Alice")


typedef struct student2{
    char *name;
    int age;
} student2;


void change(student2 *s2, char new){
    s2->name = new;
}

int main() {
    //Problem 1 a
    
    char *str = "Hello, World!";
    *str = 'h';
    /*printf("%s\n", str);
    */
    //Problem 1 c -e
    // if( fopen("a.txt", "r") == NULL){
    //     printf("error");
    // } else{
    // FILE *fp = fopen("a.txt", "r");

    // }
    //fclose(fp);

    //Question 2
    readatof("a.txt");
    student2 *s2;
    s2->name = "Bob";
    s2->age = 20; 
    char *new = "Alice"; 
    change(s2, new);

}
typedef struct node{
    int data;
    node *next;
} node;

typedef struct LL{
    node *head;
    int size;
}
