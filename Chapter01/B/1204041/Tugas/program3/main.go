package main

import "fmt"

func program3(W float64, x float64, b float64) float64 {
	// rumus z
	z := W*x + b
	return z
}

func main() {
	// definisi variabel2
	W := 2.0
	x := 3.0
	b := 1.0

	// hitung z
	z := program3(W, x, b)

	fmt.Println("z = ", z)
}
