package main

import (
	"fmt"
	"math/rand"
)

func main() {
	n := 2.1957
	W := make([][]float64, int(n)) // Matriks W berukuran n x n
	x := make([]float64, int(n))   // Vektor x berukuran n

	// Mengisi matriks W dan vektor x dengan nilai acak
	for i := 0; i < int(n); i++ {
		W[i] = make([]float64, int(n))
		for j := 0; j < int(n); j++ {
			W[i][j] = rand.Float64()
		}
		x[i] = rand.Float64()
	}

	// Menghitung z = Wx
	z := make([]float64, int(n))
	for i := 0; i < int(n); i++ {
		for j := 0; j < int(n); j++ {
			z[i] += W[i][j] * x[j]
		}
	}

	// Mencetak hasil
	fmt.Println("Matriks W:")
	for i := 0; i < int(n); i++ {
		fmt.Println(W[i])
	}
	fmt.Println("Vektor x:", x)
	fmt.Printf("z = Wx, z = %v\n", z)
}
