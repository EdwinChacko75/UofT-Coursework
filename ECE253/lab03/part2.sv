module FullAdder(
    input logic a, b, ci,     
    output logic s, co      
);

assign s = a ^ b ^ ci;
assign co = (a & b) | (ci & (a ^ b));

endmodule

module FourBitAdder(input logic [3:0] a, b, input logic c_in,
output logic [3:0] s, output logic c_out);

logic c1, c2, c3;

FullAdder FA0 (.a(a[0]), .b(b[0]), .ci(c_in), .s(s[0]), .co(c1));
FullAdder FA1 (.a(a[1]), .b(b[1]), .ci(c1),  .s(s[1]), .co(c2));
FullAdder FA2 (.a(a[2]), .b(b[2]), .ci(c2),  .s(s[2]), .co(c3));
FullAdder FA3 (.a(a[3]), .b(b[3]), .ci(c3),  .s(s[3]), .co(c_out));

endmodule

module part2 (
    input [3:0] A, B,        
    input [1:0] Function,     
    output [7:0] ALUout      
);

wire [3:0] sum_result; // 4-bit sum result
wire c_out; // carry out from FourBitAdder
reg [7:0] temp; 

FourBitAdder RCA (.a(A), .b(B), .c_in(1'b0), .s(sum_result), .c_out(c_out));

always_comb begin
    case(Function)
        2'b00: temp[7:0] = {3'b0, c_out, sum_result}; 
        2'b01: temp[7:0] = {7'b0, |(A) | |(B)}; 
        2'b10: temp[7:0] = {7'b0, &(A) & &(B)};   
        2'b11: temp[7:0] = {A, B};            
        default: temp[7:0] = 8'b00000000;            
    endcase
end

assign ALUout = temp;

endmodule
