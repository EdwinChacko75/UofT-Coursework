
module my_tff (
    input logic clk,
    input logic rst,
    input logic T,
    input logic enable,
    output logic Q,
    output logic Q_bar
);
    logic D;
    logic Q_internal;
wire k;
assign k = Q_internal;

    assign D = T & enable ? ~Q_internal : Q_internal;

 
    always_ff @(posedge clk or posedge rst) begin
        if (rst)
            Q_internal <= 0;
        else
            Q_internal <= ((~T)&k) | ((~k)&T);
    end

    assign Q = Q_internal;
    assign Q_bar = ~Q_internal;

endmodule

module part1 (
    input logic Clock,
    input logic Reset,
    input logic Enable,
    output logic [7:0] CounterValue

);

    wire q0, q1, q2, q3, q4, q5, q6, q7, w1, w2, w3, w4, w5, w6, w7;
    assign q0 = CounterValue[0];
    assign q1 = CounterValue[1];
    assign q2 = CounterValue[2];
    assign q3 = CounterValue[3];
    assign q4 = CounterValue[4];
    assign q5 = CounterValue[5];
    assign q6 = CounterValue[6];
    assign q7 = CounterValue[7];

    assign w1 = q0 & Enable;
    assign w2 = q1 & w1;
    assign w3 = q2 & w2;
    assign w4 = q3 & w3;
    assign w5 = q4 & w4;
    assign w6 = q5 & w5;
    assign w7 = q6 & w6;

    my_tff ff0 (.clk(Clock), .rst(Reset), .T(Enable), .enable(Enable), .Q(CounterValue[0]), .Q_bar());
    my_tff ff1 (.clk(Clock), .rst(Reset), .T(w1), .enable(Enable), .Q(CounterValue[1]), .Q_bar());
    my_tff ff2 (.clk(Clock), .rst(Reset), .T(w2), .enable(Enable), .Q(CounterValue[2]), .Q_bar());
    my_tff ff3 (.clk(Clock), .rst(Reset), .T(w3), .enable(Enable), .Q(CounterValue[3]), .Q_bar());
    my_tff ff4 (.clk(Clock), .rst(Reset), .T(w4), .enable(Enable), .Q(CounterValue[4]), .Q_bar());
    my_tff ff5 (.clk(Clock), .rst(Reset), .T(w5), .enable(Enable), .Q(CounterValue[5]), .Q_bar());
    my_tff ff6 (.clk(Clock), .rst(Reset), .T(w6), .enable(Enable), .Q(CounterValue[6]), .Q_bar());
    my_tff ff7 (.clk(Clock), .rst(Reset), .T(w7), .enable(Enable), .Q(CounterValue[7]), .Q_bar());

endmodule
