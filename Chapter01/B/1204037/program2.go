package main

import "fmt"

func Wx(W [][]float64, x []float64) []float64 {
	n := len(x)
	z := make([]float64, n)

	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			z[i] += W[i][j] * x[j]
		}
	}

	return z
}

func main() {
	W := [][]float64{{1.0, 2.0}, {3.0, 4.0}} // contoh matriks 2x2
	x := []float64{5.0, 6.0}                 // contoh vektor ukuran 2
	z := Wx(W, x)
	fmt.Println(z) // output: [17 39]
}
