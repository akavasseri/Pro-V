// Black-box DUT under verification. The coverage agent reads this ONLY to
// confirm the interface (port names / widths), never to decide functional tests.
module top_module(
    input  a,
    input  b,
    output y
);
    assign y = a & b;
endmodule
