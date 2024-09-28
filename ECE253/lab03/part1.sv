
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
FullAdder FA0 (.a(a[0]), .b(b[0]), .ci(c_in), .s(s[0]), .co(c_out[0]));
FullAdder FA1 (.a(a[1]), .b(b[1]), .ci(c_out[0]),  .s(s[1]), .co(c_out[1]));
FullAdder FA2 (.a(a[2]), .b(b[2]), .ci(c_out[1]),  .s(s[2]), .co(c_out[2]));
FullAdder FA3 (.a(a[3]), .b(b[3]), .ci(c_out[2]),  .s(s[3]), .co(c_out[3]));

endmodule

