package main

import (
	"fmt"
	"math"
)

func main() {
	// set nilai n
	n := 2.1957

	// bulatkan n ke bilangan bulat terdekat
	roundedN := int(math.Round(n))

	// buat matriks W dan vektor x dengan nilai acak
	W := [][]float64{{1, 2}, {3, 4}}
	x := []float64{2, 3}

	// pastikan W dan x sesuai dengan ukuran n yang telah dibulatkan
	if len(W) != roundedN || len(W[0]) != roundedN || len(x) != roundedN {
		fmt.Println("Ukuran matriks atau vektor tidak sesuai dengan nilai n yang telah dibulatkan")
		return
	}

	// hitung z menggunakan aturan perkalian matriks
	var z []float64
	for i := 0; i < roundedN; i++ {
		var sum float64
		for j := 0; j < roundedN; j++ {
			sum += W[i][j] * x[j]
		}
		z = append(z, sum)
	}

	// tampilkan hasil
	fmt.Println("z =", z)
}
