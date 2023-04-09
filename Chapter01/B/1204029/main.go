package main

import (
    "fmt"
    "math"
)

func main() {
    // Masukkan nilai x
    x := 5.0

    // Hitung nilai eksponensial dari x
    exp_x := math.Exp(x)

    // Cetak hasilnya
    fmt.Printf("Nilai eksponensial dari %.2f adalah %.2f\n", x, exp_x)
}
