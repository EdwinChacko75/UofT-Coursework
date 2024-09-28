#include <stdio.h>
#include <string.h>
#include <stdlib.h>
void set_int1(int x)
{
    x = 42;
}
void set_int2(int *p_x)
{
    *p_x = 42;
}
// in main call each function and pass an integer variable or its pointer
// the first will not work as intended as it will only change the value in the scope of the functino
// the second works since it goes to the address of the variable and changes it there, in the greater scope
typedef struct student1{
    char name[200];
    char student_number[11];
    int year;
} student1;
void set_default_name(student1 *p_s){
    strcpy((*p_s).name , "Default Name");

}
void create_block1(student1 **p_p_s, int n_students){
    *p_p_s = (student1 *)malloc(n_students*sizeof(student1));
}
void set_name(student1 *p_s1, char *input){
    strcpy(p_s1->name, input);
    p_s1->name[199] = '\0';

}
void destroy_block1(student1 *p_s){
    free(p_s);
}

typedef struct student2{
    char *name;
    char *student_number;
    int year;
} student2;
void create_block2(student2 **p_p_s, int num_students){
    (*p_p_s)->name =0;
    (*p_p_s)->student_number =0;
}
void set_name2(student2 *p_s, char *input){
    p_s->name = (char *)malloc(1000 *sizeof(char)); //strlen(input)*

}

int main(){
    student1 s1;
    student2 s2;
    strcpy(s1.name, "edwin");
    strcpy(s1.student_number, "1009149716");
    s1.year = 1;
    //set_default_name(&s1);
        // Question 2 
    //printf("%s, %s, %d\n",s1.name, s1.student_number, s1.year);
        // Question 3a
    //printf("%s",s1.name);
        // Question 3b
    //3b: it is the same as before, no effect on global scope
        // Question 4
        // Question 5
    // set_name(&s1, "0123456789");
    // printf("%s",s1.name);
        // Question 6
    // student1 *p_s1 = &s1;
    // destroy_block1(p_s1);

    student2 *p_s2 = &s2;
    char name2[] = "edwin";
    set_name2(&*p_s2,name2);


}
