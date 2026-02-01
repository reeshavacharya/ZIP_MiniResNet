import torch
import torch.nn as nn

intervals = [-10.0, 0.0]
coefficients = [[0.0, 0.0, 0.0, 0.0], [0.0, 1.0000000000000007, -1.6568799999999998e-16, 9.535999999999999e-18]]

def approx_relu(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)

    def poly_eval(z: torch.Tensor, coeffs):
        # Evaluate the polynomial sum of coeffs[k] * z^k
        result = torch.zeros_like(z)
        for power, c in enumerate(coeffs):
            result += c * (z ** power)
        return result

    num_intervals = len(intervals)
    for i in range(num_intervals - 1):
        lower = intervals[i]
        upper = intervals[i + 1]
        mask = (x >= lower) & (x < upper)
        out[mask] = poly_eval(x[mask], coefficients[i])

    # Values >= the last boundary
    mask_last = (x >= intervals[-1])
    out[mask_last] = poly_eval(x[mask_last], coefficients[-1])

    # Values < the first boundary
    mask_first = (x < intervals[0])
    out[mask_first] = poly_eval(x[mask_first], coefficients[0])

    return out
