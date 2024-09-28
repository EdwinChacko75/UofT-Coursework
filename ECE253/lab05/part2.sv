module RateDivider #(
  parameter CLOCK_FREQUENCY = 500
)(
  input logic ClockIn,
  input logic Reset,
  input logic [1:0] Speed,
  output logic Enable
);

  logic [($clog2(CLOCK_FREQUENCY * 4) - 1):0] N;
  logic [($clog2(CLOCK_FREQUENCY * 4) - 1):0] count;

  always_comb begin
    case (Speed)
      2'b00: N = 0;
      2'b01: N = CLOCK_FREQUENCY - 1;
      2'b10: N = (CLOCK_FREQUENCY * 2) - 1;
      2'b11: N = (CLOCK_FREQUENCY * 4) - 1;
      default: N = 0;
    endcase
  end

  always_ff @(posedge ClockIn or posedge Reset) begin
    if (Reset)
      count <= N;
    else if (count != 0)
      count <= count - 1;
    else
      count <= N;
    
    Enable <= (count == 0) ? 1 : 0;
  end
endmodule

module DisplayCounter (
  input logic Clock,
  input logic Reset,
  input logic EnableDC,
  output logic [3:0] CounterValue
);

  // Counter for displaying hexadecimal values
  logic [3:0] counter = 4'b0000;

  always_ff @(posedge Clock or posedge Reset) begin
    if (Reset)
      counter <= 4'b0000;
    else if (EnableDC)
      counter <= counter + 1;
  end

  assign CounterValue = counter;

endmodule

module part2
#(parameter CLOCK_FREQUENCY = 500)(
  input logic ClockIn,
  input logic Reset,
  input logic [1:0] Speed,
  output logic [3:0] CounterValue
);

  logic EnableDC;

  RateDivider #(CLOCK_FREQUENCY) rate_divider (
    .ClockIn(ClockIn),
    .Reset(Reset),
    .Speed(Speed),
    .Enable(EnableDC)
  );

  DisplayCounter display_counter (
    .Clock(ClockIn),
    .Reset(Reset),
    .EnableDC(EnableDC),
    .CounterValue(CounterValue)
  );

endmodule
