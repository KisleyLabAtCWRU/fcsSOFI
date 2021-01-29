classdef ModelID
    properties (Constant = true)
        GAUSS_1D = 0
        GAUSS_2D = 1
        GAUSS_2D_ELLIPTIC = 2
        GAUSS_2D_ROTATED = 3
        CAUCHY_2D_ELLIPTIC = 4
        LINEAR_1D = 5
        FLETCHER_POWELL = 6
        BROWN_DENNIS = 7
        EXPONENTIAL = 8
        BROWNIAN_1COMP = 9
        BROWNIAN_2COMP = 10
        ANOMALOUS = 11
        BROWNIAN_1COMP_NORM = 12
        BROWNIAN_1COMP_TAUA = 13
        ANOMALOUS_2PARAM_TAUA = 14
    end
end