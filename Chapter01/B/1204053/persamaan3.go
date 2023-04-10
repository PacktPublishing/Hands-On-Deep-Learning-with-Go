package main

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

func main() {
	// Membuat matriks W, vektor x, dan vektor bias b
	n := 2.1957
	W := mat.NewDense(2, 2, []float64{1, 2, 3, 4})
	x := mat.NewVecDense(2, []float64{n, n})
	b := mat.NewVecDense(2, []float64{5, 6})

	// Menghitung z = Wx + b
	var z mat.VecDense
	z.MulVec(W, x)
	z.AddVec(&z, b)

	// Menampilkan hasil
	fmt.Println("z = Wx + b")
	fmt.Printf("W = \n%v\n", mat.Formatted(W))
	fmt.Printf("x = %v\n", mat.Formatted(x))
	fmt.Printf("b = %v\n", mat.Formatted(b))
	fmt.Printf("z = %v\n", mat.Formatted(&z))
}
