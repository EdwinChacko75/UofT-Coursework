module part1(
    input logic Clock,
    input logic Reset,
    input logic w,
    output logic z,
    output logic [3:0] CurState
);

    typedef enum logic [3:0] {A = 4'd0, B = 4'd1, C = 4'd2, D = 4'd3, 
                              E = 4'd4, F = 4'd5, G = 4'd6} statetype;
    statetype y_Q, Y_D;

   
    always_comb begin
        case (y_Q)
            A: begin 
					if (!w) Y_D = A;
					else Y_D = B;
				end		
            B: Y_D = (w) ? C : A; 
            C: Y_D = (w) ? D : E; 
            D: Y_D = (w) ? F : E; 
            E: Y_D = (w) ? G : A; 
            F: Y_D = (w) ? F : E; 
            G: Y_D = (w) ? C : A; 
            default: Y_D = y_Q;      
        endcase
    end 

   
    always_ff @(posedge Clock) begin
        if (Reset == 1'b1)
            y_Q <= A; 
        else
            y_Q <= Y_D; 
    end 

    
    assign z = ((y_Q == F) | (y_Q == G));

    assign CurState = y_Q; 
endmodule
