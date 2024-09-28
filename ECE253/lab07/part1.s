.global _start
.text
_start:
	la s2, LIST
	addi s10, zero, 0
	addi s11, zero, 0

# first word, since not considered in loop
lw s9, 0(s2)
addi s11, s11, 1   # increment digit coutner
add s10, s10, s9 # add current word to counter

loop:
    addi s2, s2, 4 #increment address
    lw s3, 0(s2) # set current word
    # Check if null char
    li s4, -1
    beq s3, s4, END
    
    # if not null char
    addi s11, s11, 1   # increment digit coutner
    add s10, s10, s3 # add current word to counter
    

    j loop

END:
	ebreak
	
.global LIST
.data
LIST:
.word 1, 2, 3, 5, 10,-1