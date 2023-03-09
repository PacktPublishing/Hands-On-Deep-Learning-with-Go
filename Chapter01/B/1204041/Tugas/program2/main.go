package main

import "fmt"

func program2(W [2][2]float64, x [2]float64) [2]float64 {
	var z [2]float64
	z[0] = W[0][0]*x[0] + W[0][1]*x[1]
	z[1] = W[1][0]*x[0] + W[1][1]*x[1]
	return z
}

func main() {
	// Definisi matrix W dan vektor x
	W := [2][2]float64{{1.2, 3.4}, {5.6, 7.8}}
	n := 2.1957
	x := [2]float64{n, 1.0}

	// W kali x
	z := program2(W, x)

	fmt.Println("z = ", z)
}
