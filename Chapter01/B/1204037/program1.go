package main

import (
	"fmt"

	"gorgonia.org/tensor"
)

func main() {
	// Inisialisasi tensor a dan b
	a := tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float64{1, 2, 3, 4}))
	b := tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float64{5, 6, 7, 8}))

	// Lakukan operasi penjumlahan antara tensor a dan b
	c, _ := tensor.Add(a, b)

	// Tampilkan hasil penjumlahan
	fmt.Println(c)
}
