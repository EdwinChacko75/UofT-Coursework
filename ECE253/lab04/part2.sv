
module part2(
    input logic Clock,
    input logic Reset_b,
    input logic [3:0] Data,
    input logic [1:0] Function,
    output logic [7:0] ALUout);
	
    logic [7:0] reg_data;

	always_ff @(posedge Clock) 
	begin
        if (Reset_b) ALUout[7:0] <= 8'b00000000;
        else 
		begin 
			case (Function)
				2'b00: ALUout <= Data[3:0] + reg_data[3:0];
				2'b01: ALUout <= Data[3:0] * reg_data[3:0];
				2'b10: ALUout <= reg_data[3:0] << Data[3:0];
				2'b11: ALUout <= reg_data[3:0]; 
				default: ALUout <= 8'b00000000; 
			endcase
		end
    end
	assign reg_data = ALUout[7:0];
endmodule

