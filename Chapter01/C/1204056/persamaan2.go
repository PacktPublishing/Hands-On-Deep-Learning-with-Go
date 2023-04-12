package main

import (
	"fmt"
	"math"
)

func main() {

	// Inisialisasi matriks W dan vektor x
	W := [2][2]float64{{1.0, 2.0}, {3.0, 4.0}}
	x := [2]float64{5.0, 6.0}

	// Hitung z = Wx
	var z [2]float64
	for i := 0; i < 2; i++ {
		for j := 0; j < 2; j++ {
			z[i] += W[i][j] * x[j]
		}
	}

	// Tampilkan hasil
	fmt.Printf("z = [%.4f, %.4f]\n", math.Round(z[0]*10000)/10000, math.Round(z[1]*10000)/10000)
}
