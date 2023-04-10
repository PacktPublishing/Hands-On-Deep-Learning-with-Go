package main

import (
	"fmt"
)

func main() {
	// Inisialisasi matriks W dan vektor x
	W := [2][2]float64{{5.0, 6.0}, {7.0, 8.0}}
	x := [2]float64{2.1957, 1.0}

	// Hitung z = Wx
	var z [2]float64
	for i := 0; i < 2; i++ {
		for j := 0; j < 2; j++ {
			z[i] += W[i][j] * x[j]
		}
	}

	// Cetak hasil
	fmt.Println("z = ", z)
}
