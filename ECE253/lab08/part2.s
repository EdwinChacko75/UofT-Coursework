.global _start

.text

_start:
    la a5, LIST          
    lw a3, 0(a5)        
    addi a5, a5, 4       
    addi t1, zero, 1
    sub a3, a3, t1       
    li t2, 0            
    li t3, 0             

Outer_loop:
    bge t2, a3, Finish  
    li t0, 0             
    jal Inner_loop       
    beqz t0, Finish    
    addi t2, t2, 1      
    j Outer_loop        

Inner_loop:
    sub t4, a3, t2      
    bge t3, t4, Reset_inner
    lw a1, 0(a5)        
    lw a2, 4(a5)        
    bgt a1, a2, Do_swap  
    Continue_inner:
    addi t3, t3, 1      
    addi a5, a5, 4      
    j Inner_loop        

Reset_inner:
    li t3, 0           
    la a5, LIST          
    addi a5, a5, 4       
    jr ra                

Do_swap:
    mv t5, a1            
    mv a1, a2            
    mv a2, t5
    sw a1, 0(a5)         
    sw a2, 4(a5)
    li t0, 1            
    j Continue_inner 

Finish:
    ebreak               

.global LIST
.data
LIST:
    .word 10, 1400, 45, 23, 5, 3, 8, 17, 4, 20, 33
