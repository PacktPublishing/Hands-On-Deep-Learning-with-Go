package main

import "fmt"

func program3(W [][]float64, x []float64, b []float64) []float64 {
    n := len(W)
    z := make([]float64, n)
    for i := 0; i < n; i++ {
        for j := 0; j < n; j++ {
            z[i] += W[i][j] * x[j]
        }
        z[i] += b[i]
    }
    return z
}

func main() {
    W := [][]float64{{1.0, 2.0}, {3.0, 4.0}}
    x := []float64{1.0, 2.0}
    b := []float64{0.5, 0.5}
    z := program3(W, x, b)
    fmt.Printf("%v\n", z)
}
