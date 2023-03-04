package main

import (
	"fmt"
	"math/rand"
)

func main() {
	n := 2.1957

	// Membuat matriks W dan vektor x secara acak
	W := make([][]float64, int(n))
	x := make([]float64, int(n))
	for i := 0; i < int(n); i++ {
		W[i] = make([]float64, int(n))
		for j := 0; j < int(n); j++ {
			W[i][j] = rand.Float64()
		}
		x[i] = rand.Float64()
	}

	// Membuat nilai b secara acak
	b := rand.Float64()

	// Menghitung nilai z = Wx + b
	z := make([]float64, int(n))
	for i := 0; i < int(n); i++ {
		for j := 0; j < int(n); j++ {
			z[i] += W[i][j] * x[j]
		}
		z[i] += b
	}

	// Mencetak hasil
	fmt.Println("Matriks W:")
	for i := 0; i < int(n); i++ {
		fmt.Println(W[i])
	}

	fmt.Println("\nVektor x:", x)

	fmt.Println("\nb:", b)

	fmt.Println("\nz = Wx + b")
	fmt.Printf("z = %v\n", z)
}
