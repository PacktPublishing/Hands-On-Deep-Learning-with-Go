package main

import (
    "fmt"
    "math"
)

func main() {
    
    f := 2.0        // frekuensi
    a := 1.0        // amplitudo
    phi := math.Pi  // fase awal
    lambda := 0.01  // panjang interval


    start := 0.0
    end := 1.0
    for x := start; x <= end; x += lambda {
        y := a * math.Sin(2*math.Pi*f*x+phi)

        fmt.Printf("x = %v\ty = %v\n", x, y)
    }
}