`timescale 1ns / 1ns

module v7404 (
    input logic pin1, pin3, pin5, pin9, pin11, pin13,
    output logic pin2, pin4, pin6, pin8, pin10, pin12
);

    assign pin2 = ~pin1;
    assign pin4 = ~pin3;
    assign pin6 = ~pin5;
    assign pin8 = ~pin9;
    assign pin10 = ~pin11;
    assign pin12 = ~pin13;

endmodule

module v7408 (
    input logic pin1, pin2, pin4, pin5, pin9, pin10, pin12, pin13,
    output logic pin3, pin6, pin8, pin11
);

    assign pin3 = pin1 & pin2;
    assign pin6 = pin4 & pin5;
    assign pin8 = pin9 & pin10;
    assign pin11 = pin12 & pin13;

endmodule

module v7432 (
    input logic pin1, pin2, pin4, pin5, pin9, pin10, pin12, pin13,
    output logic pin3, pin6, pin8, pin11
);

    assign pin3 = pin1 | pin2;
    assign pin6 = pin4 | pin5;
    assign pin8 = pin9 | pin10;
    assign pin11 = pin12 | pin13;

endmodule

module mux(
    output logic [9:0] LEDR,
    input logic [9:0] SW
);

    mux2to1 u0(
        .x(SW[0]),
        .y(SW[1]),
        .s(SW[9]),
        .m(LEDR[0])
    );

endmodule

module mux2to1(
    input logic x, y, s,
    output logic m
);

    logic notS, sx, notSy;

    v7404 notS_inst (
        .pin1(s),
        .pin2(notS)
    );

    v7408 sx_inst (
        .pin1(s),
        .pin2(x),
        .pin3(sx)
    );

    v7408 notSy_inst (
        .pin1(notS),
        .pin2(y),
        .pin3(notSy)
    );

    v7432 m_inst (
        .pin1(sx),
        .pin2(notSy),
        .pin3(m)
    );

endmodule

