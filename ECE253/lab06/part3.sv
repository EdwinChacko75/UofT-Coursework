module part3(
    input logic Clock,
    input logic Reset,
    input logic Go,
    input logic [3:0] Divisor,
    input logic [3:0] Dividend,
    output logic [3:0] Quotient,
    output logic [3:0] Remainder,
    output logic ResultValid
);

    reg [4:0] A;
    reg [3:0] Q; 
    reg [3:0] M; 
    reg [2:0] count; 
    reg [1:0] state;

  
    typedef enum logic [1:0] {
        S_IDLE = 2,
        S_LOAD = 3,
        S_CALC = 0,
        S_DONE = 1
    } state_t;

    state_t current_state, next_state;


    logic load, shift, add, sub;
    logic [4:0] sub_result; 

    
    always @(posedge Clock or posedge Reset) begin
        if (Reset) begin
            current_state <= S_IDLE;
        end else begin
            current_state <= next_state;
        end
    end

  
    always_comb begin
        case (current_state)
            S_IDLE: next_state = Go ? S_LOAD : S_IDLE;
            S_LOAD: next_state = S_CALC;
            S_CALC: next_state = (count == 0) ? S_DONE : S_CALC;
            S_DONE: next_state = Go ? S_DONE : S_IDLE; 
            default: next_state = S_IDLE;
        endcase
    end

    
    always_comb begin
        
        load = 0;
        shift = 0;
        add = 0;
        sub = 0;
        ResultValid = 0;

        case (current_state)
            S_LOAD: begin
                load = 1;
            end
            S_CALC: begin
                shift = 1;
                if (A[4] == 0) begin
                    sub = 1; 
                end else begin
                    add = 1;
                end
            end
            S_DONE: begin
                ResultValid = 1; 
            end
        endcase
    end

    
    always_ff @(posedge Clock) begin
        if (Reset) begin
            A <= 5'b0;
            Q <= 4'b0;
            M <= 4'b0;
            count <= 3'b100; 
            sub_result <= 5'b0;
        end else begin
            if (load) begin
               
                M <= Divisor;
                Q <= 0;
                A <= {1'b0, Dividend};
            end
            if (shift) begin
               
                {A, Q} <= {A[3:0], Q, Dividend[4-count]};
            end
            if (sub) begin
                sub_result <= A - {1'b0, M}; 
                if (sub_result[4] == 0) begin
                    Q[0] <= 1;
                end
            end
            if (add) begin
                sub_result <= A + {1'b0, M}; 
            end
            if (current_state == S_CALC) begin
                if (sub_result[4] == 1) begin
                  
                    A <= sub_result + {1'b0, M};
                    Q[0] <= 0;
                end else begin
                
                    A <= sub_result;
                end
                count <= count - 1;
            end
        end
    end

   
    assign Quotient = Q;
    assign Remainder = A[3:0];
endmodule

