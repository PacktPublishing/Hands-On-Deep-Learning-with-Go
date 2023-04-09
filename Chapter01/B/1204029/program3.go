package main

import (
    "fmt"
)

func main() {
    // Inisialisasi nilai W, x, dan b
    var W float64 = 5.0
    var x float64 = 4.0
    var b float64 = 3.0

    // Hitung nilai z = Wx + b
    var z float64 = W*x + b

    // Cetak nilai z ke konsol
    fmt.Println("z =", z)
}
