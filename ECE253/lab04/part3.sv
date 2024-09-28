module submod(input logic Clock,
				input logic reset,
				input logic LoadLeft,
				input logic loadn,
				input logic D,
				input logic right,
				input logic left,
				output logic Q);

	logic w, v;

	assign w = LoadLeft ? left : right;
	assign v = loadn ? w : D;


	always_ff @(posedge Clock) 
		begin
			if (reset) Q <= 1'b0;
			else Q <= v;
		end
endmodule
 
module part3(input logic clock,
			input logic reset,
			input logic ParallelLoadn,
			input logic RotateRight,
			input logic ASRight,
			input logic [3:0] Data_IN,
			output logic [3:0] Q);
	
	logic f, g, q0, q1, q2, q3;
	
	assign q0 = Q[0];	
	assign q1 = Q[1];
	assign q2 = Q[2];
	assign q3 = Q[3];
	
	assign f = RotateRight;
	assign g = ASRight ? q0 : q3;
	
	submod u0(.reset(reset), .Clock(clock), .LoadLeft(f), .loadn(ParallelLoadn), .D(Data_IN[3]), .right(q2), .left(g), .Q(Q[3]));
	submod u1(.reset(reset), .Clock(clock), .LoadLeft(f), .loadn(ParallelLoadn), .D(Data_IN[2]), .right(q1), .left(q3), .Q(Q[2]));
	submod u2(.reset(reset), .Clock(clock), .LoadLeft(f), .loadn(ParallelLoadn), .D(Data_IN[1]), .right(q0), .left(q2), .Q(Q[1]));
	submod u3(.reset(reset), .Clock(clock), .LoadLeft(f), .loadn(ParallelLoadn), .D(Data_IN[0]), .right(q3), .left(q1), .Q(Q[0]));
	
	always_ff @(posedge clock) 
	begin
        if (reset) Q[3:0] <= 4'b0000;
	end
	
endmodule

