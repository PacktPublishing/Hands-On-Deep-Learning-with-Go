package main

import "fmt"

func main() {
	// Inisialisasi nilai untuk W, x, dan b
	var W float64 = 2.5
	var x float64 = 3.0
	var b float64 = 1.5

	// Hitung nilai z = Wx + b
	var z float64 = W*x + b

	// Tampilkan hasil perhitungan
	fmt.Printf("Nilai z = %.2f", z)
}