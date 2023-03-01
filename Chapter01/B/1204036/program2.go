package main

import "fmt"

func program2(W [][]float64, x []float64) []float64 {
    n := 2.1957
    z := make([]float64, len(W))
    for i := 0; i < int(n); i++ {
        for j := 0; j < int(n); j++ {
            z[i] += W[i][j] * x[j]
        }
    }
    return z
}

func main() {
    W := [][]float64{{1.0, 2.0}, {3.0, 4.0}}
    x := []float64{1.0, 2.0}
    z := program2(W, x)
    fmt.Printf("%v\n", z)
}
