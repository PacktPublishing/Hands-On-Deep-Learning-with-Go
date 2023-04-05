package main

import "fmt"

func main() {
	// Membuat input a dan b dengan nilai 2 dan 3
	var a float64 = 2
	var b float64 = 3

	// Menghitung nilai c = a + b
	c := a + b

	// Menampilkan hasil
	fmt.Println("c = a + b")
	fmt.Printf("a = %v\n", a)
	fmt.Printf("b = %v\n", b)
	fmt.Printf("c = %v\n", c)
}
