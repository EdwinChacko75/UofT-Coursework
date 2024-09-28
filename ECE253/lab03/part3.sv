`timescale 1ns / 1ns

module FullAdder(
    input logic a, b, ci,    
    output logic s, co     
);

assign s = a ^ b ^ ci;
assign co = (a & b) | (ci & (a ^ b));

endmodule

module part1(input logic [3:0] a, b, input logic c_in,
output logic [3:0] s, c_out);

logic c1, c2, c3;

FullAdder FA0 (.a(a[0]), .b(b[0]), .ci(c_in), .s(s[0]), .co(c1));
FullAdder FA1 (.a(a[1]), .b(b[1]), .ci(c1),  .s(s[1]), .co(c2));
FullAdder FA2 (.a(a[2]), .b(b[2]), .ci(c2),  .s(s[2]), .co(c3));
FullAdder FA3 (.a(a[3]), .b(b[3]), .ci(c3),  .s(s[3]), .co(c_out));

endmodule


module part3 #(parameter N = 4) (
    input [N-1:0] A, B,         
    input [1:0] Function,       
    output [(2*N)-1:0] ALUout   
);

reg [(2*N)-1:0] temp; 

always_comb begin
    case(Function)
        2'b00: temp[(2*N)-1:0] = A + B;
        2'b01: temp[(2*N)-1:0] = |{A, B};  
        2'b10: temp[(2*N)-1:0] = &{A, B};  
        2'b11: temp[(2*N)-1:0] = {A, B};
        default: temp[(2*N)-1:0] = {(2*N){1'b0}};
    endcase
end

assign ALUout = temp;
endmodule


