// Black-box DUT under verification (OR variant, for the AND-vs-OR contrast).
module top_module(
    input  a,
    input  b,
    output y
);
    assign y = a | b;
endmodule
