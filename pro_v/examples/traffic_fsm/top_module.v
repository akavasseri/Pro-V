// Black-box DUT under verification. Interface confirmation only.
module top_module(
    input  clk,
    input  rst,
    input  start,
    input  stall,
    output grant,
    output done
);
    // ... internal state logic is opaque to the coverage agent ...
endmodule
