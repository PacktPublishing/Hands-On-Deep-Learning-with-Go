package main

import (
	"fmt"
	"math/rand"
)

func main() {
	// Definisikan variabel dan konstanta
	var W, x [2]float64
	var b float64

	// Inisialisasi nilai untuk W, x, dan b
	for i := 0; i < 2; i++ {
		W[i] = rand.Float64()
		x[i] = rand.Float64()
	}
	b = rand.Float64()

	// Hitung nilai z
	var z float64
	for i := 0; i < 2; i++ {
		z += W[i] * x[i]
	}
	z += b

	// Cetak hasil
	fmt.Printf("W = [%f %f]\n", W[0], W[1])
	fmt.Printf("x = [%f %f]\n", x[0], x[1])
	fmt.Printf("b = %f\n", b)
	fmt.Printf("z = %f\n", z)
}
