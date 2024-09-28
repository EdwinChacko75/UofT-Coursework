module part3 #(parameter CLOCK_FREQUENCY=500)(
    input logic ClockIn,
    input logic Reset,
    input logic Start,
    input logic [2:0] Letter,
    output logic DotDashOut,
    output logic NewBitOut
);

    logic [11:0] A = 12'b101110000000;
    logic [11:0] B = 12'b110101010100;
    logic [11:0] C = 12'b110101100100;
    logic [11:0] D = 12'b110101010000;
    logic [11:0] E = 12'b101100000000;
    logic [11:0] F = 12'b101010110000;
    logic [11:0] G = 12'b110110100000;
    logic [11:0] H = 12'b101010101000;

    logic loadShiftReg = 0; 
    logic [31:0] rateDivider;
    logic [11:0] shiftReg;
    logic [1:0] state, nextState;
    logic [1:0] bitLengthCounter = 0;


    always @(posedge ClockIn or posedge Reset) begin
        if (Reset) rateDivider <= 0;
        else if (rateDivider < CLOCK_FREQUENCY/2 - 1) rateDivider <= rateDivider + 1;
        else rateDivider <= 0;
    end

   
    always @(posedge ClockIn) begin
        if (loadShiftReg) begin
            case (Letter)
                3'b000: shiftReg <= A;
                3'b001: shiftReg <= B;
                3'b010: shiftReg <= C;
                3'b011: shiftReg <= D;
                3'b100: shiftReg <= E;
                3'b101: shiftReg <= F;
                3'b110: shiftReg <= G;
                3'b111: shiftReg <= H;
            endcase
            loadShiftReg = 0;
        end
        else if (rateDivider == CLOCK_FREQUENCY/2 - 1) shiftReg <= shiftReg << 1;
    end

 
    always @(posedge ClockIn or posedge Reset) begin
        if (Reset) state <= 2'b00;
        else state <= nextState;
    end

   
    always @(posedge ClockIn or posedge Reset) begin
        if (Start && state == 2'b00) loadShiftReg = 1;
    end

  
    always @(*) begin
        case(state)
            2'b00: begin
                if (Start) nextState = 2'b01;
                else nextState = 2'b00;
            end
            2'b01: begin
                if (shiftReg[11]) begin
                    if (bitLengthCounter < 2) nextState = 2'b01;
                    else nextState = 2'b10;
                end
                else nextState = 2'b10;
            end
            2'b10: begin
                if (rateDivider == CLOCK_FREQUENCY/2 - 1) nextState = 2'b01;
            end
        endcase
    end

  
    always @(posedge ClockIn or posedge Reset) begin
        if (Reset) bitLengthCounter <= 0;
        else if (state == 2'b01) bitLengthCounter <= bitLengthCounter + 1;
        else bitLengthCounter <= 0;
    end

    always @(*) begin
        DotDashOut = (state == 2'b01) ? shiftReg[11] : 0;
        NewBitOut = (rateDivider == 0 && (state == 2'b01 || state == 2'b10));
    end

endmodule

