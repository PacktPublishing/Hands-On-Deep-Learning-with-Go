package main

import (
	"fmt"
	"math"
)

func main() {
	// input nilai jari-jari lingkaran
	var r float64 = 7

	// menghitung persamaan lingkaran
	var equation string = fmt.Sprintf("x^2 + y^2 = %.0f", math.Pow(r, 2))
	fmt.Println("Persamaan lingkaran:", equation)
}
