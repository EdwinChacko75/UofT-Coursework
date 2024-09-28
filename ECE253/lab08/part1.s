.global _start
.text

# Main program entry
_start:
    la s2, LIST      
    lw s3, 0(s2)       

    addi s1, zero, 0   

MAIN_LOOP:
    li t5, -1        
    beq s3, t5, END     
    jal ra, ONES      
    bgt a0, s10, UPDATE_MAX

    addi s2, s2, 4
    lw s3, 0(s2)
    j MAIN_LOOP         
    
UPDATE_MAX:
    add s10, a0, zero    
    j MAIN_LOOP       

END:
    ebreak      
  # Counting ones subroutine
ONES:
    addi t0, zero, 0    
    addi t1, zero, 0   
    addi t2, zero, 1  
    li t4, 1            

FIND_ONES:
    beqz s3, FINAL_CHECK 
    and t3, s3, t2     
    bnez t3, INCREMENT  
    j RESET_AND_CHECK    

INCREMENT:
    addi t0, t0, 1      
    j NEXT_BIT

RESET_AND_CHECK:
    bgt t0, t1, UPDATE_LONG
    addi t0, zero, 0      
NEXT_BIT:
    srl s3, s3, t4        
    j FIND_ONES

FINAL_CHECK:
    bgt t0, t1, UPDATE_LONG
    j RETURN_MAX

UPDATE_LONG:
    add t1, t0, zero     

RETURN_MAX:
    add a0, t1, zero  
    jr ra

# Data section
.global LIST
.data
LIST: 
.word 0x1fffffff, 0x12345678, -1, 0x7fffffff