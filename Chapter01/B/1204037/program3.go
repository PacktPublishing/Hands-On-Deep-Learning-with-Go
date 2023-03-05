package main

import "fmt"

func WxAddb(W [][]float64, x []float64, b []float64) []float64 {
	n := len(x)
	z := make([]float64, n)

	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			z[i] += W[i][j] * x[j]
		}
		z[i] += b[i]
	}

	return z
}

func main() {
	W := [][]float64{{1.0, 2.0}, {3.0, 4.0}} // contoh matriks 2x2
	x := []float64{5.0, 6.0}                 // contoh vektor ukuran 2
	b := []float64{1.0, 2.0}                 // contoh vektor bias ukuran 2
	z := WxAddb(W, x, b)
	fmt.Println(z) // output: [18 41]
}
